# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import math

from ..layers.inference import combine_for_reading_2x, \
    restore_y_2x, restore_y_2x_with_cat_after, restore_y_4x,  \
    add_and_multiply, replicate_pad, clamp_reciprocal_with_quant
from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder
from ..layers.inference import AdaptiveQuant
import torch.cuda.amp as amp

class CompressionModel(nn.Module):
    def __init__(self, z_channel, extra_qp=0):
        super().__init__()
        self.z_channel = z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(64 + extra_qp, z_channel)
        self.gaussian_encoder = GaussianEncoder()
        self.masks = {}
        self.quant_ste = AdaptiveQuant(mode="ste")
        self.quant_noise = AdaptiveQuant(mode="noise")

    @staticmethod
    def get_qp_num():
        return 64
    
    @staticmethod
    def probs_to_bits(probs):
        factor = -1.0 / math.log(2.0)
        bits = torch.log(probs + 1e-5) * factor
        bits = torch.clamp(bits, 0, None)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        # Force everything here to run in full precision
        #orig_dtype = y.dtype
        #with amp.autocast(enabled=False):
        y_f = y.to(torch.float32)
        sigma_f = sigma.to(torch.float32)

        # 1) sanitize inputs
        y_f = torch.nan_to_num(y_f, nan=0.0, posinf=1e4, neginf=-1e4)
        sigma_f = torch.nan_to_num(sigma_f, nan=1e-5, posinf=1e10, neginf=1e-5)
        sigma_f = sigma_f.clamp_(1e-5, 1e10)

        # 2) standardized bounds for [y-0.5, y+0.5]
        inv_sigma = 1.0 / sigma_f
        z_hi = (y_f + 0.5) * inv_sigma
        z_lo = (y_f - 0.5) * inv_sigma

        # 3) clamp where erf is stable
        z_hi = z_hi.clamp(-12.0, 12.0)
        z_lo = z_lo.clamp(-12.0, 12.0)

        # 4) CDF difference
        root2 = math.sqrt(2.0)
        probs = 0.5 * (torch.erf(z_hi / root2) - torch.erf(z_lo / root2))

        # 5) safety against underflow / NaNs
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp_min(1e-9)

        bits = -torch.log2(probs)

        return bits #.to(orig_dtype)
    
    def get_z_bits(self, z, bit_estimator, index):
        probs = bit_estimator.get_cdf(z + 0.5, index) - bit_estimator.get_cdf(z - 0.5, index)
        probs = probs.to(torch.float32)
        return CompressionModel.probs_to_bits(probs)
    
    def update(self, force=False):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(self.entropy_coder, force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(self.entropy_coder, force=force, entropy_coder=self.entropy_coder)

    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    @staticmethod
    def get_downsampled_shape(height, width, p):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_r, padding_b = self.get_padding_size(H, W, 4)
        y_pad = replicate_pad(y, padding_b, padding_r)
        return y_pad

    @staticmethod
    def separate_prior_for_video_encoding(params, y):
        q_dec, scales, means = params.chunk(3, 1)
        q_dec, y = clamp_reciprocal_with_quant(q_dec, y, 0.5)
        return y, q_dec, scales, means
    

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q_hat, y_q_hat_write = self.quant_ste(y_res) * mask, self.quant_noise(y_res) * mask
        # y_q = torch.round(y_res)
        y_hat = y_q_hat + means_hat

        return y_res, y_q_hat, y_q_hat_write, y_hat, scales_hat

    @staticmethod
    def get_one_mask(micro_mask, height, width, dtype, device):
        mask = torch.tensor(micro_mask, dtype=dtype, device=device)
        mask = mask.repeat((height + 1) // 2, (width + 1) // 2)
        mask = mask[:height, :width]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        return mask

    def get_mask_2x(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_2x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 2 == 0
                m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 1)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (1, 0)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1), dim=1)
                mask_1 = torch.cat((m * m1, m * m0), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @staticmethod
    def single_part_for_writing_2x(x):
        x0, x1 = x.chunk(2, 1)
        return x0 + x1

    def compress_prior_2x(self, y, common_params, y_spatial_prior, write=False, fm_s=None):
        y, q_dec, scales, means = self.separate_prior_for_video_encoding(common_params, y)
        # --- FM modulation (optional) ---
        if fm_s is not None:
            # Expect fm_s shape [1 or B, C, 1, 1]; will broadcast over H,W
            y      = y / fm_s
            scales = scales / fm_s
            means  = means  / fm_s
        # ---------------------------------

        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)
        with torch.no_grad():
            assert torch.allclose(mask_0 + mask_1, torch.ones_like(mask_0)), "mask_0 and mask_1 not complementary!"

        y_res_0, y_q_hat_0, y_q_hat_write_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        scales, means = y_spatial_prior(cat_params).chunk(2, 1)
        y_res_1, y_q_hat_1, y_q_hat_write_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_res = y_res_0 + y_res_1
        y_q_hat = y_q_hat_0 + y_q_hat_1
        y_q_hat_write = y_q_hat_write_0 + y_q_hat_write_1
        scales_hat = s_hat_0 + s_hat_1

        y_hat = add_and_multiply(y_hat_0, y_hat_1, q_dec)

        # --- FM symmetry after dequant ---
        if fm_s is not None:
            y_hat = y_hat * fm_s
        # ---------------------------------

        if write:
            y_q_w_0 = self.single_part_for_writing_2x(y_q_hat_write_0)
            y_q_w_1 = self.single_part_for_writing_2x(y_q_hat_write_1)
            s_w_0 = self.single_part_for_writing_2x(s_hat_0)
            s_w_1 = self.single_part_for_writing_2x(s_hat_1)
            return y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat

        return y_res, y_q_hat, y_q_hat_write, y_hat, scales_hat

    ##############for iamge model##################
    def get_mask_4x(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_4x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 0)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (0, 0)), height, width, dtype, device)
                m2 = self.get_one_mask(((0, 0), (1, 0)), height, width, dtype, device)
                m3 = self.get_one_mask(((0, 0), (0, 1)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]
    
    def separate_prior(self, params, is_video=False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            quant_step = torch.clamp_min(quant_step, 0.5)
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    @staticmethod
    def single_part_for_writing_4x(x):
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)
    
    def compress_prior_4x(self, y, common_params, y_spatial_prior_reduction,
                          y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                          y_spatial_prior_adaptor_3, y_spatial_prior, write=False, fm_s=None):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        common_params:(_,N*2+2,_,_)
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params, False)
        if fm_s is not None:
            # Fold FM into the prior-coded latent path
            y      = y / fm_s
            scales = scales / fm_s
            means  = means  / fm_s
            # (q_enc isn’t used explicitly here; we multiply back via q_dec below)

        common_params = y_spatial_prior_reduction(common_params)   #(_,N,_,_)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)
        with torch.no_grad():
            assert torch.allclose(mask_0 + mask_1 + mask_2 + mask_3, torch.ones_like(mask_0)), "Masks are not fully complementary (some overlap or missing)"

        y = y * q_enc

        y_res_0, y_q_hat_0, y_q_hat_write_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_res_1, y_q_hat_1, y_q_hat_write_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_res_2, y_q_hat_2, y_q_hat_write_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_res_3, y_q_hat_3, y_q_hat_write_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_res = (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q_hat = (y_q_hat_0 + y_q_hat_1) + (y_q_hat_2 + y_q_hat_3)
        y_q_hat_write = (y_q_hat_write_0 + y_q_hat_write_1) + (y_q_hat_write_2 + y_q_hat_write_3)
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        if fm_s is not None:
            y_hat = y_hat * fm_s


        if write:
            y_q_w_0 = self.single_part_for_writing_4x(y_q_hat_write_0)
            y_q_w_1 = self.single_part_for_writing_4x(y_q_hat_write_1)
            y_q_w_2 = self.single_part_for_writing_4x(y_q_hat_write_2)
            y_q_w_3 = self.single_part_for_writing_4x(y_q_hat_write_3)
            s_w_0 = self.single_part_for_writing_4x(s_hat_0)
            s_w_1 = self.single_part_for_writing_4x(s_hat_1)
            s_w_2 = self.single_part_for_writing_4x(s_hat_2)
            s_w_3 = self.single_part_for_writing_4x(s_hat_3)
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat
        return y_res, y_q_hat, y_q_hat_write, y_hat, scales_hat

############for decompress##############
    # @staticmethod
    # def separate_prior_for_video_decoding(params):
    #     quant_step, scales, means = params.chunk(3, 1)
    #     quant_step = torch.clamp_min(quant_step, 0.5)
    #     return quant_step, scales, means

    # def decompress_prior_2x(self, common_params, y_spatial_prior):
    #     infos = self.decompress_prior_2x_part1(common_params)
    #     y_hat = self.decompress_prior_2x_part2(common_params, y_spatial_prior, infos)
    #     return y_hat

    # def decompress_prior_2x_part1(self, common_params):
    #     q_dec, scales, means = self.separate_prior_for_video_decoding(common_params)
    #     dtype = means.dtype
    #     device = means.device
    #     B, C, H, W = means.size()
    #     mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

    #     scales_r = combine_for_reading_2x(scales, mask_0, inplace=False)
    #     indexes, skip_cond = self.gaussian_encoder.build_indexes_decoder(scales_r)
    #     self.gaussian_encoder.decode_y(indexes)
    #     infos = {
    #         "q_dec": q_dec,
    #         "mask_0": mask_0,
    #         "mask_1": mask_1,
    #         "means": means,
    #         "scales_r": scales_r,
    #         "skip_cond": skip_cond,
    #         "indexes": indexes,
    #     }
    #     return infos

    # def decompress_prior_2x_part2(self, common_params, y_spatial_prior, infos):
    #     dtype = common_params.dtype
    #     device = common_params.device
    #     y_q_r = self.gaussian_encoder.get_y(infos["scales_r"].shape,
    #                                         infos["scales_r"].numel(),
    #                                         dtype, device,
    #                                         infos["skip_cond"], infos["indexes"])
    #     y_hat_0, cat_params = restore_y_2x_with_cat_after(y_q_r, infos["means"], infos["mask_0"],
    #                                                       common_params)
    #     scales, means = y_spatial_prior(cat_params).chunk(2, 1)
    #     scales_r = combine_for_reading_2x(scales, infos["mask_1"], inplace=True)
    #     y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
    #     y_hat_1 = restore_y_2x(y_q_r, means, infos["mask_1"])

    #     y_hat = add_and_multiply(y_hat_0, y_hat_1, infos["q_dec"])
    #     return y_hat

####################for key_frame################################

    # def decompress_prior_4x(self, common_params, y_spatial_prior_reduction,
    #                         y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
    #                         y_spatial_prior_adaptor_3, y_spatial_prior):
    #     _, quant_step, scales, means = self.separate_prior(common_params, False)
    #     common_params = y_spatial_prior_reduction(common_params)
    #     dtype = means.dtype
    #     device = means.device
    #     B, C, H, W = means.size()
    #     mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

    #     scales_r = self.single_part_for_writing_4x(scales * mask_0)
    #     y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
    #     y_hat_curr_step = restore_y_4x(y_q_r, means, mask_0)
    #     y_hat_so_far = y_hat_curr_step

    #     params = torch.cat((y_hat_so_far, common_params), dim=1)
    #     scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
    #     scales_r = self.single_part_for_writing_4x(scales * mask_1)
    #     y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
    #     y_hat_curr_step = restore_y_4x(y_q_r, means, mask_1)
    #     y_hat_so_far = y_hat_so_far + y_hat_curr_step

    #     params = torch.cat((y_hat_so_far, common_params), dim=1)
    #     scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
    #     scales_r = self.single_part_for_writing_4x(scales * mask_2)
    #     y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
    #     y_hat_curr_step = restore_y_4x(y_q_r, means, mask_2)
    #     y_hat_so_far = y_hat_so_far + y_hat_curr_step

    #     params = torch.cat((y_hat_so_far, common_params), dim=1)
    #     scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
    #     scales_r = self.single_part_for_writing_4x(scales * mask_3)
    #     y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
    #     y_hat_curr_step = restore_y_4x(y_q_r, means, mask_3)
    #     y_hat_so_far = y_hat_so_far + y_hat_curr_step

    #     y_hat = y_hat_so_far * quant_step

    #     return y_hat

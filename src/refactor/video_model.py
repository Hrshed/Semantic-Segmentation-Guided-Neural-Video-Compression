# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""pixel-shuffle + config.py
"""
import torch
import torch.nn.functional as F
from torch import nn

from .common_model import CompressionModel
from ..layers.layers import SubpelConv2x, DepthConvBlock, \
    ResidualBlockUpsample, ResidualBlockWithStride2
from .config import DMCConfig

class FeatureExtractor(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),) 
        
        self.conv2 = nn.Sequential(
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),)

    def forward(self, x, quant):
        x1, ctx_t = self.forward_part1(x, quant)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def forward_part1(self, x, quant):
        x1 = self.conv1(x)
        ctx_t = x1 * quant
        return x1, ctx_t

    def forward_part2(self, x1):
        ctx = self.conv2(x1)
        return ctx


class Encoder(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.conv1 = nn.Conv2d(cfg.src, cfg.ch_d, 1)
        self.conv2 = nn.Sequential(
            DepthConvBlock(cfg.ch_d * 2, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d))
        
        self.down = nn.Conv2d(cfg.ch_d, cfg.ch_y, 3, stride=2, padding=1)

    def forward(self, x, ctx, quant_step):
        x = F.pixel_unshuffle(x, self.patch_size)
        x = self.conv1(x)
        x = self.conv2(torch.cat((x, ctx), 1))
        x = x * quant_step
        return self.down(x)

class Decoder(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.up   = SubpelConv2x(cfg.ch_y, cfg.ch_d, 3, padding=1)
        self.conv = nn.Sequential(
            DepthConvBlock(cfg.ch_d * 2, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),)
        
        self.proj = nn.Conv2d(cfg.ch_d, cfg.ch_d, 1)

    def forward(self, x, ctx, quant_step):
        x = self.up(x)
        x = x * quant_step
        x = self.conv(torch.cat((x, ctx), 1))
        return self.proj(x)

class ReconGeneration(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.conv = nn.Sequential(
            DepthConvBlock(cfg.ch_d, cfg.ch_recon),
            DepthConvBlock(cfg.ch_recon, cfg.ch_recon),
            DepthConvBlock(cfg.ch_recon, cfg.ch_recon),
            DepthConvBlock(cfg.ch_recon, cfg.ch_recon),)
        
        self.head = nn.Conv2d(cfg.ch_recon, cfg.src, 1)

    def forward(self, x, quant_step):
        x = self.conv(x)
        x = x * quant_step
        x = self.head(x)
        x = F.pixel_shuffle(x, self.patch_size)
        return torch.clamp(x, 0.0, 1.0)

class HyperEncoder(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(cfg.ch_y, cfg.ch_z),
            ResidualBlockWithStride2(cfg.ch_z, cfg.ch_z),
            ResidualBlockWithStride2(cfg.ch_z, cfg.ch_z),)

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(cfg.ch_z, cfg.ch_z),
            ResidualBlockUpsample(cfg.ch_z, cfg.ch_z),
            DepthConvBlock(cfg.ch_z, cfg.ch_y),)

    def forward(self, x):
        return self.conv(x)


class PriorFusion(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(cfg.ch_y * 3, cfg.ch_y * 3),
            DepthConvBlock(cfg.ch_y * 3, cfg.ch_y * 3),
            DepthConvBlock(cfg.ch_y * 3, cfg.ch_y * 3),
            nn.Conv2d(cfg.ch_y * 3, cfg.ch_y * 3, 1),)

    def forward(self, x):
        return self.conv(x)


class SpatialPrior(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(cfg.ch_y * 4, cfg.ch_y * 3),
            DepthConvBlock(cfg.ch_y * 3, cfg.ch_y * 3),
            nn.Conv2d(cfg.ch_y * 3, cfg.ch_y * 2, 1),)

    def forward(self, x):
        return self.conv(x)

class DMC(CompressionModel):
    def __init__(self, cfg: DMCConfig):
        super().__init__(z_channel=cfg.ch_z, extra_qp=cfg.extra_qp)
        self.qp_shift = cfg.qp_shift
        self.cfg = cfg

        # modules
        self.feature_adaptor_i = DepthConvBlock(cfg.src, cfg.ch_d)
        self.feature_adaptor_p = nn.Conv2d(cfg.ch_d, cfg.ch_d, 1)
        self.feature_extractor = FeatureExtractor(cfg)
        self.encoder           = Encoder(cfg)
        self.hyper_encoder     = HyperEncoder(cfg)
        self.hyper_decoder     = HyperDecoder(cfg)
        self.temporal_prior_encoder = ResidualBlockWithStride2(cfg.ch_d, cfg.ch_y * 2)
        self.y_prior_fusion    = PriorFusion(cfg)
        self.y_spatial_prior   = SpatialPrior(cfg)
        self.decoder           = Decoder(cfg)
        self.recon_generation_net = ReconGeneration(cfg)

        self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_d, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_d, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_d, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_recon, 1, 1)))

    def res_prior_param_decoder(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        params = self.y_prior_fusion(
            torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, q_recon):
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature, q_recon)
        return x_hat, feature

    def shift_qp(self, qp, fa_idx):
        return qp + self.qp_shift[fa_idx]
    
    def forward(self, x, qp, dpb, after_i=True):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        qp = torch.tensor([qp], device=x.device)

        q_encoder = self.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]
        q_recon   = self.q_recon[qp:qp+1, :, :, :]
        
        if after_i:
            feature = self.feature_adaptor_i(F.pixel_unshuffle(dpb["frame"], 8))
        else:
            feature = self.feature_adaptor_p(dpb["feature"])

        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        y = self.encoder(x, ctx, q_encoder)

        hyper_inp = self.pad_for_y(y)
        z = self.hyper_encoder(hyper_inp)
        z_hat = self.quant_ste(z)
        z_hat_write = self.quant_noise(z)

        params = self.res_prior_param_decoder(z_hat, ctx_t)

        _, _, y_q_hat_write, y_hat, scales_hat = self.compress_prior_2x(
            y, params, self.y_spatial_prior, write=False)

        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)

        _, _, H, W = x.size()
        pixel_num = H * W
        y_for_bit = y_q_hat_write
        z_for_bit = z_hat_write

        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z, qp)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp = bpp_y + bpp_z

        if x.is_cuda:
            torch.cuda.synchronize(device=x.device)

        return {
            "dpb": {"frame": x_hat, "feature": feature},
            "bpp":  bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }
    
if __name__ == "__main__":
    model = DMC(DMCConfig()).cuda()
    dummy = torch.rand(1, 3, 256, 256, device="cuda")
    dpb  = {"frame": dummy, "feature": None}
    print(model(dummy, dpb=dpb, qp=32))
    
    for k, v in model.named_parameters():
        print(f"{k}, shape:{v.shape}")

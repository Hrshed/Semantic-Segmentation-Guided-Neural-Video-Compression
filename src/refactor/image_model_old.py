"""pixel-shuffle + recon
"""
import torch
from torch import nn
import torch.nn.functional as F


from .common_model import CompressionModel
from ..layers.layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2

g_ch_src = 3 * 8 * 8
g_ch_enc_dec = 192
g_ch_recon = 64

class IntraEncoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.enc_1 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_src, g_ch_enc_dec))
            
        self.enc_2 = nn.Conv2d(g_ch_enc_dec, N, 3, stride=2, padding=1)

    def forward(self, x, quant_step):
        out = F.pixel_unshuffle(x, 8)
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.dec_1 = ResidualBlockUpsample(N, g_ch_enc_dec)

        self.dec_2 = nn.Sequential( 
            DepthConvBlock(g_ch_enc_dec, g_ch_src),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),)

    def forward(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        out = self.dec_2(out)
        return out

class ReconGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec,     g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
        )
        self.head = nn.Conv2d(g_ch_recon, g_ch_src, 1)

    def forward(self, x, quant_step):
        out = self.conv(x)
        out = out * quant_step
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out
    

class DMCI(CompressionModel):
    def __init__(self, N=128, z_channel=64):
        super().__init__(z_channel=z_channel)

        self.enc = IntraEncoder(N)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
        )

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel),
            ResidualBlockUpsample(z_channel, z_channel),
            DepthConvBlock(z_channel, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2 + 2, 1),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2, 1),
        )

        self.dec = IntraDecoder(N)
        self.recon_generation_net = ReconGeneration()

        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_recon, 1, 1)))

    def forward(self, x, qp):
        device = x.device  
        qp = torch.tensor([qp], device=x.device)

        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]
        curr_q_recon = self.q_recon[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)
        y_pad = self.pad_for_y(y)

        z = self.hyper_enc(y_pad)
        z_hat = self.quant_ste(z)
        z_hat_write = self.quant_noise(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)

        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        (
            _, _, y_q_hat_write, y_hat, scales_hat
        ) = self.compress_prior_4x(
            y,
            params,
            self.y_spatial_prior_reduction,
            self.y_spatial_prior_adaptor_1,
            self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3,
            self.y_spatial_prior,
            write=False
        )

        feature = self.dec(y_hat, curr_q_dec)
        x_hat = self.recon_generation_net(feature, curr_q_recon)

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
            torch.cuda.synchronize(device=device)

        return {
            "dpb": {"frame": x_hat, "feature": feature},
            "bpp":  bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }
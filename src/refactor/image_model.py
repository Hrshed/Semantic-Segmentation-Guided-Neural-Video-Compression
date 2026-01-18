"""pixel-shuffle + config.py
"""
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

from .common_model import CompressionModel
from ..layers.layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from .config import DMCIConfig

class IntraEncoder(nn.Module):
    def __init__(self, cfg: DMCIConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.enc_1 = nn.Sequential(
            DepthConvBlock(cfg.src, cfg.enc_dec),
            DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
            DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
            DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
            DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
            DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
            DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
        )

        self.enc_2 = nn.Conv2d(cfg.enc_dec, cfg.N, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, quant_step: torch.Tensor) -> torch.Tensor:
        out = F.pixel_unshuffle(x, self.patch_size)
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, cfg: DMCIConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.dec_1 = ResidualBlockUpsample(cfg.N, cfg.enc_dec)

        self.dec_2 = nn.Sequential(DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.enc_dec),
                                   DepthConvBlock(cfg.enc_dec, cfg.src))

    def forward(self, x: torch.Tensor, quant_step: torch.Tensor) -> torch.Tensor:
        out = self.dec_1(x)
        out = out * quant_step
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, self.patch_size)
        return out
   

class DMCI(CompressionModel):
    def __init__(self, cfg : DMCIConfig):
        super().__init__(z_channel=cfg.z_channel)
        self.cfg = cfg

        self.enc = IntraEncoder(cfg)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock(cfg.N, cfg.z_channel),
            ResidualBlockWithStride2(cfg.z_channel, cfg.z_channel),
            ResidualBlockWithStride2(cfg.z_channel, cfg.z_channel),
        )

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(cfg.z_channel, cfg.z_channel),
            ResidualBlockUpsample(cfg.z_channel, cfg.z_channel),
            DepthConvBlock(cfg.z_channel, cfg.N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(cfg.N, cfg.N * 2),
            DepthConvBlock(cfg.N * 2, cfg.N * 2),
            DepthConvBlock(cfg.N * 2, cfg.N * 2),
            nn.Conv2d(cfg.N * 2, cfg.N * 2 + 2, 1),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(cfg.N * 2 + 2, cfg.N, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(cfg.N * 2, cfg.N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(cfg.N * 2, cfg.N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(cfg.N * 2, cfg.N * 2, force_adaptor=True)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(cfg.N * 2, cfg.N * 2),
            DepthConvBlock(cfg.N * 2, cfg.N * 2),
            DepthConvBlock(cfg.N * 2, cfg.N * 2),
            nn.Conv2d(cfg.N * 2, cfg.N * 2, 1),)

        self.dec = IntraDecoder(cfg)

        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), cfg.enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), cfg.enc_dec, 1, 1)))

    def forward(self, x, qp):
        device = x.device  
        qp = torch.tensor([qp], device=x.device)

        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

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

        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)

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
            "dpb": {"frame": x_hat, "feature": None},
            "bpp":  bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }
    

# ```python
if __name__ == "__main__":


    model = DMCI(DMCIConfig()).cuda()
    print(model(torch.rand(1,3,256,256).cuda(), qp=32))
    for k, v in model.named_parameters():
        print(f"{k}, shape:{v.shape}")
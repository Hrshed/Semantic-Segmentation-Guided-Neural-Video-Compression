import torch
import torch.nn.functional as F
from torch import nn

from .common_model import CompressionModel
from ..layers.layers import SubpelConv2x, DepthConvBlock, \
    ResidualBlockUpsample, ResidualBlockWithStride2
from .config import DMCConfig
from .mask_predictor import MaskPredictor


class FeatureExtractor(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
        )

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
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
        )
        self.down = nn.Conv2d(cfg.ch_d, cfg.ch_y, 3, stride=2, padding=1)

    def forward(self, x, ctx, quant_step):
        # x is expected to be 3ch YCbCr here (mask, if present, is not used by the encoder)
        x = F.pixel_unshuffle(x, self.patch_size)
        x = self.conv1(x)
        x = self.conv2(torch.cat((x, ctx), 1))
        x = x * quant_step
        return self.down(x)


class Decoder(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.up = SubpelConv2x(cfg.ch_y, cfg.ch_d, 3, padding=1)
        self.conv = nn.Sequential(
            DepthConvBlock(cfg.ch_d * 2, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
        )
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
            DepthConvBlock(cfg.ch_recon, cfg.ch_recon),
        )
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
            ResidualBlockWithStride2(cfg.ch_z, cfg.ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(cfg.ch_z, cfg.ch_z),
            ResidualBlockUpsample(cfg.ch_z, cfg.ch_z),
            DepthConvBlock(cfg.ch_z, cfg.ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class PriorFusion(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(cfg.ch_y * 3, cfg.ch_y * 3),
            DepthConvBlock(cfg.ch_y * 3, cfg.ch_y * 3),
            DepthConvBlock(cfg.ch_y * 3, cfg.ch_y * 3),
            nn.Conv2d(cfg.ch_y * 3, cfg.ch_y * 3, 1),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPrior(nn.Module):
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(cfg.ch_y * 4, cfg.ch_y * 3),
            DepthConvBlock(cfg.ch_y * 3, cfg.ch_y * 3),
            nn.Conv2d(cfg.ch_y * 3, cfg.ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)

def _finite_check(t: torch.Tensor, tag: str):
    if not torch.isfinite(t).all():
        tmin = float(torch.nanmin(t).item())
        tmax = float(torch.nanmax(t).item())
        raise RuntimeError(f"[NaNGuard] non-finite activations after {tag} (min={tmin}, max={tmax})")
    

class SFT(nn.Module):
    """
    Spatial Feature Transform for mask.

    Architecture mirrors Encoder:
        conv1 -> 3x DepthConvBlock -> down

    Differences vs Encoder:
        - input:  patch_size^2 channels (mask after F.pixel_unshuffle)
        - output: 2 * ch_d channels (split into gamma, beta)
        - QP-conditioned via a per-QP scaling embedding.
    """
    def __init__(self, cfg: DMCConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.ch_d = cfg.ch_d

        in_ch = self.patch_size * self.patch_size  # 8*8 for mask after pixel_unshuffle

        self.conv1 = nn.Conv2d(in_ch, cfg.ch_d, kernel_size=1)
        self.conv2 = nn.Sequential(
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
            DepthConvBlock(cfg.ch_d, cfg.ch_d),
        )
        self.down = nn.Conv2d(cfg.ch_d, cfg.ch_y * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, qp) -> tuple[torch.Tensor, torch.Tensor]:

        x = F.pixel_unshuffle(x, self.patch_size)
        x = self.conv1(x)     
        x = self.conv2(x)
        x = x * qp     
        x = self.down(x)          

        gamma, beta = x.chunk(2, dim=1) 

        return gamma, beta








class DMC(CompressionModel):
    def __init__(self, cfg: DMCConfig):
        super().__init__(z_channel=cfg.ch_z, extra_qp=cfg.extra_qp)
        self.qp_shift = cfg.qp_shift
        self.cfg = cfg

        # modules
        self.feature_adaptor_i = DepthConvBlock(cfg.src, cfg.ch_d)
        self.feature_adaptor_p = nn.Conv2d(cfg.ch_d, cfg.ch_d, 1)
        self.feature_extractor = FeatureExtractor(cfg)
        self.encoder = Encoder(cfg)
        self.hyper_encoder = HyperEncoder(cfg)
        self.hyper_decoder = HyperDecoder(cfg)
        self.temporal_prior_encoder = ResidualBlockWithStride2(cfg.ch_d, cfg.ch_y * 2)
        self.y_prior_fusion = PriorFusion(cfg)
        self.y_spatial_prior = SpatialPrior(cfg)
        self.decoder = Decoder(cfg)
        self.recon_generation_net = ReconGeneration(cfg)

        # Old mask adaptor using pooling
        self.hyper_in_adapter = nn.Conv2d(cfg.ch_y + 1, cfg.ch_y, kernel_size=1, bias=True)
        #self._init_hyper_in_adapter_identity()

        # New mask adaptor using Spatial Feature Modulation
        self.mask_sft = SFT(cfg)

        # per-QP learnable scales
        self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_d, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_d, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_d, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_recon, 1, 1)))

        # per-QP SFT (gamma/beta) parameters, same indexing style as q_encoder, etc.
        self.q_sft = nn.Parameter(
            torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_d, 1, 1))
        )


    def res_prior_param_decoder(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder(z_hat)
        _finite_check(hierarchical_params, "hierarchical_params")
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _finite_check(temporal_params, "temporal_prior_encoder")
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        _finite_check(hierarchical_params, "hierarchical_params_2")
        params = self.y_prior_fusion(torch.cat((hierarchical_params, temporal_params), dim=1))
        _finite_check(params, "y_prior_fusion")
        return params

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, q_recon):
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature, q_recon)
        return x_hat, feature

    def shift_qp(self, qp, fa_idx):
        return qp + self.qp_shift[fa_idx]

    def _prepare_hyper_input(self, y: torch.Tensor, mask_img: torch.Tensor | None):
        """
        Downsample mask to y's spatial size via average pooling (semantically correct for area),
        pad mask identically to y, then FiLM-modulate y and return y_mod for the hyper-encoder.
        Output shape matches the original hyper path: (B, ch_y, H_pad, W_pad).
        """
        # y: (B, ch_y, H_y, W_y)
        B, ch_y, H_y, W_y = y.shape

        # Pad y the same way as before (e.g., to /4 multiple)
        y_pad = self.pad_for_y(y)  # -> (B, ch_y, H_pad, W_pad)
        _, _, H_pad, W_pad = y_pad.shape

        # Prepare mask at y resolution
        if mask_img is None:
            m_down = torch.zeros(B, 1, H_y, W_y, dtype=y.dtype, device=y.device)
        else:
            # Use average pooling semantics (preserves coverage meaning when downsampling)
            # Adaptive version handles any size mismatches robustly.
            m_down = F.adaptive_avg_pool2d(mask_img.to(dtype=y.dtype), (H_y, W_y))
            m_down.clamp_(0.0, 1.0)

    

        # gamma, beta from the conv stack
        gamma, beta = self.mask_sft(m_down, q_sft)   # or self.mask_film(m_down) if you kept that name
        y_mod = y_pad * (1.0 + gamma) + beta   # same shape as y_pad


        # Sanity guard
        if not torch.isfinite(y_mod).all():
            mmin = float(torch.nanmin(y_mod).item())
            mmax = float(torch.nanmax(y_mod).item())
            raise RuntimeError(f"[NaNGuard] y_mod has non-finite values (min={mmin}, max={mmax})")

        return y_mod


    def forward(self, x, qp, dpb, after_i=True):
        # x can be (B,3,H,W) or (B,4,H,W) with last channel as mask
        if x.size(1) > 3:
            mask_img = x[:, 3:4, :, :]
            x_img = x[:, :3, :, :]
        else:
            mask_img = torch.zeros_like(x[:, :1, :, :])
            x_img = x

        qp = torch.tensor([qp], device=x.device)

        q_encoder = self.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]
        q_recon   = self.q_recon[qp:qp+1, :, :, :]
        q_sft = self.q_sft[qp:qp+1, :, :, :]

        if after_i:
            feature = self.feature_adaptor_i(F.pixel_unshuffle(dpb["frame"], 8))
        else:
            feature = self.feature_adaptor_p(dpb["feature"])


        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        y = self.encoder(x_img, ctx, q_encoder)

        gamma, beta = self.mask_sft(mask_img, q_sft)
        y = y * (1.0 + gamma) + beta   


        z = self.hyper_encoder(y)
        z_hat = self.quant_ste(z)
        z_hat_write = self.quant_noise(z)


        params = self.res_prior_param_decoder(z_hat, ctx_t)
        _, _, y_q_hat_write, y_hat, scales_hat = self.compress_prior_2x(
            y, params, self.y_spatial_prior, write=False
        )
      

        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)

        _, _, H, W = x_img.size()
        pixel_num = H * W
        y_for_bit = y_q_hat_write
        y_for_bit = y_for_bit.clamp_(-6.0, 6.0)  # keeps z-scores in a sane range
        z_for_bit = z_hat_write

        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z, qp)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp = bpp_y + bpp_z

        if x_img.is_cuda:
            torch.cuda.synchronize(device=x_img.device)

        return {
            "dpb": {"frame": x_hat, "feature": feature},
            "bpp":  bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }


if __name__ == "__main__":
    model = DMC(DMCConfig()).cuda()
    dummy_rgb = torch.rand(1, 3, 256, 256, device="cuda")
    dummy_msk = torch.zeros(1, 1, 256, 256, device="cuda")
    dpb  = {"frame": dummy_rgb, "feature": None}

    print("Test 3ch input:")
    print(model(dummy_rgb, dpb=dpb, qp=32))
    print("Test 4ch input (RGB+Mask):")
    print(model(torch.cat([dummy_rgb, dummy_msk], 1), dpb=dpb, qp=32))

    for k, v in model.named_parameters():
        print(f"{k}, shape:{v.shape}")

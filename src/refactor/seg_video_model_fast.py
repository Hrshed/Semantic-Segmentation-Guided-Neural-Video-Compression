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
    

class MaskFiLM(nn.Module):
    """
    Produce per-location scale (gamma) and shift (beta) for y using only the mask.
    Very lightweight: 3x3 -> ReLU -> 1x1 mapping to 2*ch_y, then tanh clamp.
    """
    def __init__(self, ch_y: int, mid: int = 16, gamma_max: float = 0.5, beta_max: float = 0.25):
        super().__init__()
        self.gamma_max = float(gamma_max)
        self.beta_max  = float(beta_max)
        self.net = nn.Sequential(
            nn.Conv2d(1, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2 * ch_y, kernel_size=1, padding=0),
        )

    def forward(self, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # m: (B,1,H,W) in [0,1]
        gb = self.net(m)                             # (B, 2*ch_y, H, W)
        gamma, beta = gb.chunk(2, dim=1)            # (B,ch_y,H,W) each
        gamma = gamma
        beta  = beta
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
        self._init_hyper_in_adapter_identity()

        # New mask adaptor using Spatial Feature Modulation
        self.mask_film = MaskFiLM(cfg.ch_y)

        # per-QP learnable scales
        self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_d, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_d, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_d, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((self.get_qp_num() + cfg.extra_qp, cfg.ch_recon, 1, 1)))

        '''
        # NEW: Feature modulation (FM) parameters for the latent path
        self.fm_min_y = nn.Parameter(torch.ones(1, self.cfg.ch_y, 1, 1))  # init 1.0
        self.fm_max_y = nn.Parameter(torch.ones(1, self.cfg.ch_y, 1, 1))  # init 1.0
        '''


    def _fm_scale_from_qp(self, qp):
        """
        Robust FM: positive, ordered, finite scales; log-interpolated.
        qp: scalar (int/float/tensor)
        returns: s with shape [1(or B), C, 1, 1]
        """
        Q = self.get_qp_num() + getattr(self.cfg, "extra_qp", 0)
        t = torch.as_tensor(qp, dtype=self.fm_min_y.dtype, device=self.fm_min_y.device)
        t = t.clamp(0, Q - 1) / (Q - 1)
        t = t.view(-1, 1, 1, 1)

        # 1) sanitize params (replace NaN/±inf, then clamp to positive)
        min_v = torch.nan_to_num(self.fm_min_y, nan=1.0, posinf=1e3, neginf=1e-6).clamp_min(1e-6)
        max_v = torch.nan_to_num(self.fm_max_y, nan=1.0, posinf=1e3, neginf=1e-6).clamp_min(1e-6)

        # 2) ensure ordering (max >= min + eps)
        eps = 1e-6
        max_v = torch.maximum(max_v, min_v + eps)

        # 3) log-interpolation
        log_min = torch.log(min_v)
        log_max = torch.log(max_v)
        s = torch.exp(log_min + t * (log_max - log_min))

        # 4) final safety (bound the dynamic range)
        #s = torch.nan_to_num(s, nan=1.0, posinf=1e3, neginf=1e-6).clamp(1e-6, 1e3)
        s = torch.nan_to_num(s, nan=1.0, posinf=10.0, neginf=1e-2).clamp(1e-2, 10.0)

        return s


    def _init_hyper_in_adapter_identity(self):
        """Initialize the 1×1 adapter as identity on y and zero on mask (backward compatible)."""
        w = self.hyper_in_adapter.weight.data
        b = self.hyper_in_adapter.bias.data
        w.zero_()
        b.zero_()
        # set weight[:, :, 0, 0] to identity on first ch_y channels
        ch_y = self.cfg.ch_y
        for i in range(ch_y):
            w[i, i, 0, 0] = 1.0  # y_i -> y_i
        # mask channel (index ch_y) starts with zero contribution

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

        # Pad mask **identically** to y_pad’s size
        pad_r, pad_b = self.get_padding_size(H_y, W_y, p=4)  # matches pad_for_y logic
        if pad_r or pad_b:
            # zero outside the image is more faithful for "unknown / out of frame"
            m_down = F.pad(m_down, (0, pad_r, 0, pad_b), mode="constant", value=0.0)

        # FiLM: per-location modulation of y by the (padded) mask
        gamma, beta = self.mask_film(m_down)   # (B, ch_y, H_pad, W_pad) each
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
            mask_img = None
            x_img = x

        qp = torch.tensor([qp], device=x.device)

        q_encoder = self.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]
        q_recon   = self.q_recon[qp:qp+1, :, :, :]

        if after_i:
            feature = self.feature_adaptor_i(F.pixel_unshuffle(dpb["frame"], 8))
        else:
            if dpb["feature"] is None:
                print(f"[DEBUG] dpb['feature'] is None during {'training' if self.training else 'eval'} at frame {after_i}")
                import pdb; pdb.set_trace()

            feature = self.feature_adaptor_p(dpb["feature"])

        _finite_check(feature, "feature_adaptor_i")

        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        _finite_check(ctx, "feature_extractor.ctx")
        _finite_check(ctx_t, "feature_extractor.ctx_t")

        y = self.encoder(x_img, ctx, q_encoder)
        _finite_check(y, "encoder")

        current_mask = mask_img
        
        hyper_in = self._prepare_hyper_input(y, current_mask)
        z = self.hyper_encoder(hyper_in)
        _finite_check(z, "hyper_encoder")

        z_hat = self.quant_ste(z)
        _finite_check(z_hat, "z_hat")
        z_hat_write = self.quant_noise(z)
        _finite_check(z_hat_write, "z_hat_write")

        params = self.res_prior_param_decoder(z_hat, ctx_t)

        """OLD:"""
        _, _, y_q_hat_write, y_hat, scales_hat = self.compress_prior_2x(
            y, params, self.y_spatial_prior, write=False
        )
        '''
        # NEW: apply feature modulation (FM) to the latent path
        fm_s = self._fm_scale_from_qp(qp)  # qp is already an argument to forward(...)
        _, _, y_q_hat_write, y_hat, scales_hat = self.compress_prior_2x(
            y, params, self.y_spatial_prior, write=False, fm_s=fm_s
        )
        '''

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
            "mask_pred": current_mask if not after_i else None,
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

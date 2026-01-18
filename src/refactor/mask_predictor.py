import torch
import torch.nn.functional as F
from torch import nn
from .config import DMCConfig
from ..layers.layers import WSiLU

class MaskPredictor(nn.Module):
    """
    Predicts the next segmentation mask using a compact fully convolutional network.
    Always returns a mask of the same spatial size as `prev_mask`.
    """
    def __init__(self, cfg: DMCConfig, activation=WSiLU()):
        super().__init__()
        ch_ctx = cfg.ch_d
        mid_ch = cfg.ch_d // 4  # reduce to save parameters

        self.mask_embed = nn.Conv2d(1, ch_ctx, kernel_size=3, padding=1)

        self.net = nn.Sequential(
            nn.Conv2d(3 * ch_ctx, mid_ch, kernel_size=3, padding=1),
            activation,
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            activation,
            nn.Conv2d(mid_ch, 1, kernel_size=1)
        )

    def forward(self, prev_mask: torch.Tensor, ctx: torch.Tensor, ctx_t: torch.Tensor) -> torch.Tensor:
        if prev_mask is None:
            return None

        B, _, H_mask, W_mask = prev_mask.shape
        _, _, H_feat, W_feat = ctx.shape

        # Downsample mask to feature map size for processing
        m_down = F.interpolate(prev_mask, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
        m_feat = self.mask_embed(m_down)

        # Concat with features and process
        fused = torch.cat([m_feat, ctx, ctx_t], dim=1)
        logits = self.net(fused)  # (B,1,H_feat,W_feat)

        # Upsample back to match prev_mask size
        if (H_feat, W_feat) != (H_mask, W_mask):
            logits = F.interpolate(logits, size=(H_mask, W_mask), mode='bilinear', align_corners=False)

        return logits  # same size as prev_mask






if __name__ == "__main__":
    from .config import DMCConfig
    cfg = DMCConfig()
    predictor = MaskPredictor(cfg).cuda()
    prev_mask = torch.rand(1, 1, 256, 256).cuda()
    ctx = torch.rand(1, cfg.ch_d, 256, 256).cuda()
    ctx_t = torch.rand(1, cfg.ch_d, 256, 256).cuda()

    pred_mask = predictor(prev_mask, ctx, ctx_t)
    print(pred_mask.shape)  # (1,1,256,256)
    print(torch.sigmoid(pred_mask).min(), torch.sigmoid(pred_mask).max())

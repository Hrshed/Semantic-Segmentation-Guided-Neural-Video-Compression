# compression_training/utils/compression.py
import math
import torch


YCBCR_WEIGHTS = {
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    """Compute PSNR from MSE value."""
    mse_tensor = torch.tensor(mse)
    psnr = 10 * torch.log10(max_val ** 2 / (mse_tensor + 1e-12))
    return psnr.item()


def compute_lambda(qp: int, device: torch.device, lambda_min: float = 1.0, 
                  lambda_max: float = 768.0, q_levels: int = 64):
    """Compute lambda value for rate-distortion loss."""
    val = math.exp(math.log(lambda_min) +
                   qp / (q_levels - 1) * (math.log(lambda_max) - math.log(lambda_min)))
    return torch.tensor(val, device=device, dtype=torch.float32)


def ycbcr2rgb(ycbcr):
    """Convert YUV/YCbCr to RGB for visualization."""
    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=-3)
    rgb = torch.clamp(rgb, 0., 1.)
    return rgb


def prepare_image_for_logging(img_tensor):
    """Prepare image tensor for TensorBoard logging."""
    if img_tensor is None:
        return None
    img = img_tensor.detach().cpu()
    if img.shape[0] != 3:
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        else:
            return None
    img = torch.clamp(img, 0.0, 1.0)
    return img


def get_cosine_lr(it: int, total_iter: int, base_lr: float, min_lr: float, warmup_iters: int = 0):
    """Cosine annealing learning rate schedule with warmup."""
    if warmup_iters > 0 and it < warmup_iters:
        return base_lr * it / warmup_iters

    decay_ratio = (it - warmup_iters) / max(1, total_iter - warmup_iters)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)
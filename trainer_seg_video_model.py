# trainer_video_model.py

import os
DEBUG = False

if DEBUG:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"


import json, time, math
import csv
import random
import math
from math import floor
import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import torch
if DEBUG:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.utilities import rank_zero_only

from src.refactor.config import DMCConfig
from src.refactor.image_model import DMCIConfig
from src.models.image_model import DMCI   # image model from DCVC-RT
#from src.refactor.image_model import DMCI   # updated image model
from src.models.video_model import DMC as DMC_OLD              # old baseline, no DMCConfig
from src.refactor.seg_video_model import DMC as DMC_PERF  # performance variant
from src.refactor.seg_video_model_fast import DMC as DMC_FAST  # fast variant
from src.refactor.mask_prop_seg_video_model import DMC as DMC_MASK_PROP  # mask_prop variant
from src.utils.common import get_state_dict
from omegaconf import OmegaConf, DictConfig

from src.dataset.seg_waymo_dataset_lightning import WaymoDataModule
from src.utils.build_cache import build_cache


try:
    from timm.optim.lion import Lion
except ImportError:
    Lion = None
    print("Warning: timm's Lion optimizer not found. 'lion' optimizer option will not work.")

torch.set_float32_matmul_precision("medium")

YCBCR_WEIGHTS = {"ITU-R_BT.709": (0.2126, 0.7152, 0.0722)}
CONSTRAINT_OPT = False
MASK_TRAIN = False

def ycbcr2rgb(ycbcr):
    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=-3)
    rgb = torch.clamp(rgb, 0., 1.)
    return rgb

# GOP index & weights (unchanged)
index_map = [0, 1, 0, 2, 0, 2, 0, 2]
weights_map = {0: 0.5, 1: 1.2, 2: 0.9} 

class DebugProbe:
    # --- in __init__ ---
    def __init__(self, enabled=True, save_dir="./debug_dumps", log_every=1,
                conv_hooks=True, save_bad_batch=True, run_cpu_check=False, every_n_steps: int = 1):
        self.enabled = enabled
        self.save_dir = save_dir
        self.log_every = log_every
        self.conv_hooks = conv_hooks
        # rename flag to avoid shadowing the method
        self.save_bad_batch_flag = save_bad_batch
        self.run_cpu_check = run_cpu_check
        os.makedirs(save_dir, exist_ok=True)
        self._hooks_registered = False
        self._first_conv_fail = None
        self.every_n_steps = max(1, int(every_n_steps))
        self._printed_backend_flags = False


    # ---------- low-level utilities ----------
    @staticmethod
    def _shape(t):
        return tuple(int(x) for x in t.shape)
    
    def _should(self, step: int) -> bool:
        return bool(self.enabled and (step % self.every_n_steps == 0))

    def _log(self, msg):
        if self.enabled:
            print(f"[DBG] {msg}", flush=True)

    def _finite(self, name, t: torch.Tensor):
        if not torch.is_tensor(t): return
        if t.dtype.is_floating_point and not torch.isfinite(t).all():
            bad = t[~torch.isfinite(t)]
            raise RuntimeError(f"{name} has non-finite values "
                               f"(min={float(torch.nanmin(t))}, max={float(torch.nanmax(t))}, "
                               f"example={bad.flatten()[:5].detach().cpu().tolist()})")

    def scalar_stats(self, name, t: torch.Tensor):
        if not torch.is_tensor(t): return
        if t.dtype.is_floating_point:
            return dict(
                min=float(t.detach().min()),
                max=float(t.detach().max()),
                mean=float(t.detach().mean()))
        return {}

    # ---------- Conv hooks ----------
    def _conv_fw_hook(self, name, m: nn.Conv2d):
        def hook(mod, inp, out):
            x = inp[0]
            N, Cin, Hin, Win = map(int, x.shape)
            kh, kw = (m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size))
            sh, sw = (m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride))
            ph, pw = (m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding))
            dh, dw = (m.dilation if isinstance(m.dilation, tuple) else (m.dilation, m.dilation))
            Hout = math.floor((Hin + 2*ph - dh*(kh-1) - 1)/sh) + 1
            Wout = math.floor((Win + 2*pw - dw*(kw-1) - 1)/sw) + 1
            kdim = N * max(Hout, 0) * max(Wout, 0)
            self._log(f"CONV {name}: in={self._shape(x)} -> out≈(N,{m.out_channels},{Hout},{Wout}) "
                    f"| kdim={kdim} | groups={m.groups} | contig={x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last)} | dtype={x.dtype}")
        return hook


    def _conv_bw_hook(self, name):
        def hook(mod, gin, gout):
            for g in gin:
                if g is not None and g.dtype.is_floating_point and not torch.isfinite(g).all():
                    self._first_conv_fail = name
                    raise RuntimeError(f"Non-finite gradient into {name}")
        return hook
    
    # --- broaden hook registration to ConvTranspose2d too (optional) ---
    def register_conv_hooks(self, model: nn.Module, limit=12):
        if self._hooks_registered or not self.conv_hooks or not self.enabled:
            return
        count = 0
        for n, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.register_forward_pre_hook(self._conv_pre_hook(n, m))
                m.register_forward_hook(self._conv_fw_hook(n, m))
                m.register_full_backward_hook(self._conv_bw_hook(n))
                self._log(f"Hooked {m.__class__.__name__}: {n}")
                count += 1
                if limit and count >= limit:
                    break
        self._hooks_registered = True



    # ---------- Pre-backward guard & dump ----------
    def before_backward(self, loss, scalars: dict, tensors: dict, meta: dict):
        if not self.enabled: return
        # check finite on critical tensors
        self._finite("loss", loss)
        for k, v in tensors.items():
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                self._finite(k, v)

        # print compact scalar stats periodically
        if (meta.get("global_step", 0) % self.log_every) == 0:
            stat_line = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in scalars.items()}
            self._log("SCALARS " + json.dumps(stat_line))

    def save_bad_batch(self, frames, masks, dpb: dict, extras: dict):
        if not self.enabled or not self.save_bad_batch_flag:   # <-- use _flag
            return
        safe = {"frames": frames.detach().cpu(), "masks": masks.detach().cpu()}
        # keep only a couple of dpb tensors to avoid huge files
        if isinstance(dpb, dict):
            for k in list(dpb.keys())[:4]:
                v = dpb[k]
                if torch.is_tensor(v):
                    safe[f"dpb_{k}"] = v.detach().cpu()
        ts = int(time.time())
        path = os.path.join(self.save_dir, f"bad_batch_{ts}.pt")
        torch.save({"data": safe, "extras": extras}, path)
        self._log(f"Saved failing batch to {path}")

    # ---------- Optional CPU check for one step ----------
    def cpu_check_once(self, model_step_fn, batch_dict):
        if not (self.enabled and self.run_cpu_check): return
        self._log("Running CPU check for one step...")
        try:
            model_step_fn(device="cpu", **batch_dict)
            self._log("CPU step OK (no crash).")
        except Exception as e:
            self._log(f"CPU step FAILED: {e}")

    def _tensor_stats(self, t: torch.Tensor):
        t = t.detach()
        return (
            float(t.min()),
            float(t.max()),
            float(t.mean()),
            bool(torch.isfinite(t).all()),
        )

    # --- replace your _conv_pre_hook with this stronger version ---
    def _conv_pre_hook(self, name, m: nn.Conv2d):
        def hook(mod, inp):
            x = inp[0]

            # 1) surface any async error as early as possible
            try:
                if x.is_cuda:
                    torch.cuda.synchronize()
            except Exception:
                # even synchronize can throw if a prior kernel died
                raise RuntimeError(f"Prior CUDA kernel failed before {name}")

            # 2) one-time backend flags (helps confirm cuDNN state on *this* path)
            if not self._printed_backend_flags:
                import torch as _torch
                self._log(f"BACKEND cudnn.enabled={_torch.backends.cudnn.enabled}, "
                        f"benchmark={_torch.backends.cudnn.benchmark}, "
                        f"deterministic={_torch.backends.cudnn.deterministic}")
                self._printed_backend_flags = True

            # 3) stats (your existing logging)
            xmin, xmax, xmean, xfinite = self._tensor_stats(x)
            wmin, wmax, wmean, wfinite = self._tensor_stats(mod.weight)
            bstats = ""
            if mod.bias is not None:
                bmin, bmax, bmean, bfinite = self._tensor_stats(mod.bias)
                bstats = f" | bias[min={bmin:.2e},max={bmax:.2e},ok={bfinite}]"
            self._log(
                f"PRE {name}: x[min={xmin:.2e},max={xmax:.2e},ok={xfinite}] | "
                f"W[min={wmin:.2e},max={wmax:.2e},ok={wfinite}]{bstats}"
            )

            # 4) hard guards
            if not xfinite or not wfinite:
                raise RuntimeError(f"NaN/Inf detected before {name}")

            # allow either NCHW or NHWC(channels_last) but insist on contiguity in one
            if not (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last)):
                raise RuntimeError(f"{name}: non-contiguous input (stride={tuple(x.stride())})")

            if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise RuntimeError(f"{name}: unexpected dtype {x.dtype}")

            # check groups/Cin consistency (catches subtle grouped/depthwise mismatches)
            N, Cin, Hin, Win = map(int, x.shape)
            if mod.groups > 1:
                if Cin % mod.groups != 0:
                    raise RuntimeError(f"{name}: Cin={Cin} not divisible by groups={mod.groups}")
                # weight shape is [Cout, Cin/groups, kh, kw]
                expected_cin_per_group = Cin // mod.groups
                if int(mod.weight.shape[1]) != expected_cin_per_group:
                    raise RuntimeError(
                        f"{name}: weight Cin/group={int(mod.weight.shape[1])} "
                        f"!= Cin/groups={expected_cin_per_group}"
                    )

            # 5) predicted out size sanity (you already do this in fw hook; keep it here too)
            kh, kw = (mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size))
            sh, sw = (mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride))
            ph, pw = (mod.padding if isinstance(mod.padding, tuple) else (mod.padding, mod.padding))
            dh, dw = (mod.dilation if isinstance(mod.dilation, tuple) else (mod.dilation, mod.dilation))
            Hout = math.floor((Hin + 2*ph - dh*(kh-1) - 1)/sh) + 1
            Wout = math.floor((Win + 2*pw - dw*(kw-1) - 1)/sw) + 1
            if Hout <= 0 or Wout <= 0:
                raise RuntimeError(f"{name}: zero/negative output spatial size (Hout={Hout}, Wout={Wout}).")

            # 6) excessive magnitude guard
            THRESH = 1e6
            if max(abs(xmin), abs(xmax), abs(wmin), abs(wmax)) > THRESH:
                raise RuntimeError(f"{name}: |tensor|>1e6 detected (exploding activations/weights)")
        return hook

    def model_param_norm(self, model: nn.Module) -> float:
        s = 0.0
        for p in model.parameters():
            if p is not None and p.dtype.is_floating_point:
                s += float(p.detach().float().norm().item() ** 2)
        return math.sqrt(s)

    def model_grad_norm(self, model: nn.Module) -> float:
        s = 0.0
        for p in model.parameters():
            if p is not None and p.grad is not None and p.grad.dtype.is_floating_point:
                s += float(p.grad.detach().float().norm().item() ** 2)
        return math.sqrt(s)
    
    def _grad_norm(self, params, norm_type: float = 2.0) -> float:
        # robust grad norm that ignores None grads
        total = 0.0
        for p in params:
            if p is not None and p.grad is not None:
                # make sure we measure unscaled FP32 grads
                g = p.grad.detach()
                if g.dtype != torch.float32:
                    g = g.float()
                total += float(g.norm(norm_type).item() ** 2)
        return total ** 0.5
    
    @torch.no_grad()
    def _grad_param_stats(self, model) -> dict:
        grad_sq = 0.0
        grad_max = 0.0
        param_sq = 0.0
        param_max = 0.0
        nan_grad = False
        nan_param = False
        n_params = 0
        n_with_grad = 0

        for p in model.parameters():
            if p is None or p.numel() == 0:
                continue
            n_params += p.numel()

            # parameter stats
            pdata = p.detach()
            if torch.isnan(pdata).any() or torch.isinf(pdata).any():
                nan_param = True
            # L2^2 accumulates safely in Python float
            param_sq += (pdata.float().pow(2).sum()).item()
            pm = pdata.abs().max().item()
            if pm > param_max:
                param_max = pm

            # gradient stats (if exists)
            g = p.grad
            if g is None:
                continue
            n_with_grad += 1
            if torch.isnan(g).any() or torch.isinf(g).any():
                nan_grad = True
            grad_sq += (g.detach().float().pow(2).sum()).item()
            gm = g.detach().abs().max().item()
            if gm > grad_max:
                grad_max = gm

        grad_l2 = math.sqrt(grad_sq) if grad_sq > 0.0 else 0.0
        param_l2 = math.sqrt(param_sq) if param_sq > 0.0 else 0.0

        return {
            "grad_l2": grad_l2,
            "grad_max": grad_max,
            "param_l2": param_l2,
            "param_max": param_max,
            "nan_grad": nan_grad,
            "nan_param": nan_param,
            "params_total": n_params,
            "params_with_grad": n_with_grad,
        }

    def log_grads_and_params(self, model, step: int, tag: str = ""):
        """One-stop logger for grad + param norms in your existing [DBG] format."""
        if not self._should(int(step)):
            return
        try:
            stats = self._grad_param_stats(model)
            payload = {"tag": tag}
            payload.update(stats)
            print(f"[DBG] GRADS {json.dumps(payload)}")
        except Exception as e:
            # Never crash training because of debug prints
            print(f"[DBG] GRADS_ERROR {repr(e)}")


# ============================================================================ 

# ---------------------- Config dataclasses (unchanged) ----------------------

@dataclass
class OptimizerConfig:
    optimizer_type: str = "adamw"  # "adam", "adamw", "lion"
    base_lr: float = 1e-4
    min_lr: float = 1e-5
    aux_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_iters: int = 0

@dataclass
class CompressionConfig:
    lambda_min: float = 1.0
    lambda_max: float = 768.0
    q_levels: int = 64
    index_map: List[int] = field(default_factory=lambda: [0, 1, 0, 2, 0, 2, 0, 2])
    weights_map: Dict[int, float] = field(default_factory=lambda: {0: 0.5, 1: 1.2, 2: 0.9})

@dataclass
class DatasetConfig:
    dataset_type: str = "waymo"
    data_dir: str = "./dataset/waymo"   # folder containing *.tfrecord
    batch_size: int = 1
    num_workers: int = 8
    n_frames: int = 8
    seq_len: Optional[int] = None
    slide: int = 1
    crop: Any = field(default_factory=lambda: [256, 256])
    crop_size: Optional[int] = None
    yuv_format: str = "444"
    train_val_test_split: Tuple[float, float, float] = (0.8, 0.2, 0.0)
    generate_split: bool = False
    train_split: float = 0.8
    use_cache: bool = True
    video_dir: str = ""
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None


@dataclass
class VideoCompressionConfig:
    epochs: int = 50
    dtype: str = "float32"
    accumulation_steps: int = 1
    grad_clip: float = 5.0

    log_interval: int = 50
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    save_top_k: int = 3

    out_dir: str = "out"
    image_checkpoint_path: str = "./checkpoints/cvpr2025_image.pth.tar"
    video_checkpoint_path: str = ""
    psnrm_target_path: Optional[str] = "psnrm_csv/psnrm_target.csv"
    psnrm_default_db: float = 35.0
    dmc_variant: str = "performance"
    build_cache: bool = True 


    exp_name: str = "video-compression-waymo"
    log_dir: str = "./logs"
    seed: int = 17
    precision: str = "32-true"
    num_gpus: int = 1
    resume_from_checkpoint: Optional[str] = None

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)

    # --- WMSE constraint (tau) + dual update hyperparams ---
    wmse_target: float = 1.0e-3 
    lagr_lr: float = 5e-3
    lagr_momentum: float = 0.99
    lagr_lambda_max: float = 1e3
    lagr_init_lambda: float = 1e3

    lagr_rho: float = 3.0         
    lagr_ema_alpha: float = 0.05    
    lagr_init_lambda: float = 1.0  
    lagr_lambda_max: float = 1e3
    alm_penalty_scale: float = 3.0

# --------------------------- Lightning Module ---------------------------

class VideoCompressionTrainer(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))

        self.i_frame_model = DMCI()
        variant = getattr(self.config, "dmc_variant", "performance")
        global MASK_PROP
        MASK_PROP = False 

        if variant == "old":
            self.p_frame_model = DMC_OLD()
        elif variant == "performance":
            self.p_frame_model = DMC_PERF(DMCConfig())
        elif variant == "fast":
            self.p_frame_model = DMC_FAST(DMCConfig())
        elif variant == "mask_prop":
            self.p_frame_model = DMC_MASK_PROP(DMCConfig())
            MASK_PROP = True
        else:
            raise ValueError(
                f"Unknown dmc_variant={variant!r}. Expected one of "
                "['old', 'performance', 'fast', 'mask_prop']"
            )

        self.csv_log_dir = None
        self.train_csv_path = None
        self.val_csv_path = None
        self.train_headers_written = False
        self.val_headers_written = False

        self.step_count = 0
        self.automatic_optimization = False
        self._batches_seen = 0
        self.mask_fg_weight = float(getattr(config, "mask_fg_weight", 4.0))  # >1.0 prioritizes mask==1


        self.index_map = self.config.compression.index_map
        self.weights_map = self.config.compression.weights_map
        self.lambda_min = self.config.compression.lambda_min
        self.lambda_max = self.config.compression.lambda_max
        self.q_levels = self.config.compression.q_levels

        # WMSE-only constraint dual state (keep λ ≥ 0 via log-param)
        #self.mu_wmse  = torch.tensor(math.log(self.config.lagr_init_lambda), dtype=torch.float32)
        #self.buf_wmse = torch.tensor(0.0, dtype=torch.float32)   # momentum buffer

        lam_max = float(self.config.lagr_lambda_max)         # e.g., 1e3
        lam_min = 1e-12                                      # optional floor to avoid -inf
        init_lam = float(self.config.lagr_init_lambda)       # e.g., 1e3
        init_mu  = math.log(max(init_lam, lam_min))          # μ = log λ

        self.register_buffer("mu_log", torch.tensor(init_mu, dtype=torch.float32))
        self.register_buffer("mu_mom", torch.tensor(0.0, dtype=torch.float32))  # momentum buffer
        self.register_buffer("ema_D",  torch.tensor(0.0, dtype=torch.float32))  # EMA of D

        self.wmse_target = float(self.config.wmse_target)    # tau
        self.mu_lr   = float(self.config.lagr_lr)            # step size for μ
        self.mu_beta = float(self.config.lagr_momentum)      # momentum (0.99)
        self.lam_max = lam_max
        self.lam_min = lam_min
        self.dual_update_every = 10

        # Dual Ascend buffers on device
        self.register_buffer(
            "mu_wmse",
            torch.tensor(math.log(self.config.lagr_init_lambda), dtype=torch.float32)
        )
        self.register_buffer(
            "buf_wmse",
            torch.tensor(0.0, dtype=torch.float32)
        )
        self.register_buffer("p_quantile",
            torch.tensor(float(getattr(self.config, "wmse_p_init", 0.90)), dtype=torch.float32))
        self.p_min  = float(getattr(self.config, "wmse_p_min", 0.50))
        self.p_max  = float(getattr(self.config, "wmse_p_max", 0.99))
        self.p_up   = float(getattr(self.config, "wmse_p_step_up", 0.002))
        self.p_down = float(getattr(self.config, "wmse_p_step_down", 0.010))
        self.alm_penalty_scale = float(getattr(self.config, "alm_penalty_scale", 3.0))

        # === Augmented Lagrangian state: inequality h(θ) = ROI_MSE - τ(QP) <= 0 ===
        self.register_buffer("alm_mu",  torch.zeros(1, dtype=torch.float32))  # dual μ ≥ 0
        self.register_buffer("alm_rho", torch.tensor(float(self.config.lagr_rho), dtype=torch.float32))
        self.register_buffer("alm_h_accum", torch.tensor(0.0, dtype=torch.float32))  # running mean of h
        self.register_buffer("alm_h_count", torch.tensor(0.0, dtype=torch.float32))
        # Build per-QP PSNRm schedule (blue curve)
        self._init_psnrm_schedule()


        # Debug
        self.debug = DebugProbe(
            enabled=False,
            save_dir=os.path.join(getattr(self.config, "out_dir", "./out"), "debug"),
            log_every=1,
            conv_hooks=True,
            save_bad_batch=True,
            run_cpu_check=False,  # set True if you want the CPU one-step test
        )

        #self._freeze_backbone_except_probe()

    def _is_probe_param(self, name: str) -> bool:
        return ("mask_sft" in name) or ("q_sft" in name) or ("mask_predictor" in name)

    def _split_probe_backbone(self):

        backbone_params, probe_params, aux_params = [], [], []

        for name, p in self.p_frame_model.named_parameters():
            if not p.requires_grad:
                continue

            if "bit_estimator" in name:
                aux_params.append(p)
            elif self._is_probe_param(name):
                probe_params.append(p)
            else:
                backbone_params.append(p)

        return backbone_params, probe_params, aux_params


    @staticmethod
    def _mse_from_psnr_db(psnr_db: float, max_val: float = 1.0) -> float:
        return float((max_val ** 2) / (10.0 ** (psnr_db / 10.0)))

    @staticmethod
    def _psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        return 10.0 * torch.log10(torch.tensor(max_val ** 2, device=mse.device, dtype=mse.dtype) /
                                (mse + 1e-12))

    def _init_psnrm_schedule(self):
        """
        Create self.psnrm_targets[0..63] in dB from CSV (qp, psnrm_db). Missing QPs are
        linearly interpolated; otherwise default to config.psnrm_default_db.
        """
        import csv, os
        default_db = float(getattr(self.config, "psnrm_default_db", 35.0))
        self.register_buffer("psnrm_targets", torch.full((64,), default_db, dtype=torch.float32))
        path = getattr(self.config, "psnrm_target_path", None)
        if not path or not os.path.exists(path):
            if self.global_rank == 0:
                print(f"[PSNRm] CSV not found; using default {default_db} dB for all QPs.")
            return

        pairs = []
        with open(path, "r") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                qp = row.get("qp") or row.get("QP") or row.get("q") or row.get("index")
                ps = row.get("psnrm_db") or row.get("psnr_db") or row.get("PSNRm") or row.get("psnr")
                if qp is None or ps is None:
                    continue
                qp, ps = int(qp), float(ps)
                if 0 <= qp <= 63:
                    pairs.append((qp, ps))
        if not pairs:
            if self.global_rank == 0:
                print(f"[PSNRm] No rows parsed from CSV; using default {default_db} dB.")
            return

        pairs.sort()
        # write known points
        for qp, ps in pairs:
            self.psnrm_targets[qp] = ps
        # interpolate gaps (left/right neighbors)
        xs = [qp for qp, _ in pairs]
        for q in range(64):
            if self.psnrm_targets[q].item() == default_db and q not in xs:
                left = max([x for x in xs if x <= q], default=None)
                right = min([x for x in xs if x >= q], default=None)
                if left is None and right is not None:
                    self.psnrm_targets[q] = self.psnrm_targets[right]
                elif right is None and left is not None:
                    self.psnrm_targets[q] = self.psnrm_targets[left]
                elif left is not None and right is not None and right != left:
                    w = (q - left) / (right - left)
                    self.psnrm_targets[q] = (1 - w) * self.psnrm_targets[left] + w * self.psnrm_targets[right]

    def _psnrm_target_for_qp(self, qp_eff: int) -> float:
        qp_eff = int(max(0, min(63, qp_eff)))
        return float(self.psnrm_targets[qp_eff].item())

    def _roi_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None or mask.sum() == 0:
            return F.mse_loss(pred, target, reduction="mean")
        m = (mask > 0).float().expand_as(pred)  # (B,3,H,W)
        # weight = mask → mean over ROI (official 2025 semantics)
        return F.mse_loss(pred, target, reduction="mean", weight=m)


    def _alm_ineq_term(self, h: torch.Tensor) -> torch.Tensor:
        """
        Augmented-Lagrangian term for inequality h(θ) <= 0:
        ( max(0, μ + ρ h)^2 - μ^2 ) / (2ρ)
        Returns a scalar; keep gradient through h.
        """
        h = h.view([]) if h.numel() == 1 else h.mean()
        t = torch.clamp(self.alm_mu + self.alm_rho * h, min=0.0)
        return (t.pow(2) - self.alm_mu.pow(2)) / (2.0 * self.alm_rho)
    def _alm_term_from_g(self, g: torch.Tensor, eps: float = 0.0005) -> torch.Tensor:
        """
        Dead-zone quadratic penalty:
        penalty = (ρ/2) * relu(g + eps)^2
        - Zero when g <= -eps (constraint satisfied with margin).
        - Positive when violated; grows smoothly from 0.
        - Never negative. No dual drift.
        """
        g = g.view([]) if g.numel() == 1 else g.mean()
        ge = g + eps
        gp = F.relu(ge)
        return 0.5 * self.alm_rho * gp.pow(2)

    @torch.no_grad()
    def _alm_dual_update(self):
        """μ ← [ μ + ρ * mean(h) ]_+  using accumulated violations since last step."""
        if self.alm_h_count.item() > 0:
            g_bar = self.alm_h_accum / self.alm_h_count.clamp_min(1.0)
            self.alm_mu.add_(self.alm_rho * g_bar)
            self.alm_mu.clamp_(min=0.0)
            self.alm_h_accum.zero_()
            self.alm_h_count.zero_()


    # === IMPORTANT: drop mask channel here so model always sees 3-ch YCbCr ===
    def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
        proj, frames = batch  # frames: (B, T, C, H, W)
        if isinstance(frames, torch.Tensor) and frames.ndim == 5 and frames.size(2) > 3:
            mask = frames[:, :, 3:4, ...]     # (B,T,1,H,W) in {0,1}
            frames = frames[:, :, :3, ...]    # keep Y,Cb,Cr
        else:
            # no mask provided -> use ones (no change to loss)
            B, T, _, H, W = frames.shape
            mask = torch.ones(B, T, 1, H, W, dtype=frames.dtype, device=frames.device)
        return (proj, frames, mask)

    def setup(self, stage=None):
        if stage == "fit":
            self._load_checkpoints()
            # self.load_p_from_lightning_ckpt()
            self._setup_csv_logging()

            if self.global_rank == 0:
                print("Model Structure:")
                print(f"I-frame model: {self.i_frame_model}")
                print(f"P-frame model: {self.p_frame_model}")
                print(f"Compression config: lambda_min={self.lambda_min}, lambda_max={self.lambda_max}, q_levels={self.q_levels}")
                print(f"Index map: {self.index_map}")
                print(f"Weights map: {self.weights_map}")

    @staticmethod
    def _resolve_module(root: nn.Module, dotted: str) -> nn.Module:
        """Resolve 'a.b.0.c' into a submodule object."""
        mod = root
        for name in dotted.split('.'):
            if name.isdigit():
                mod = mod._modules[name]
            else:
                mod = getattr(mod, name)
        return mod
    
    @staticmethod
    def _resolve_module(root: nn.Module, dotted: str) -> nn.Module:
        mod = root
        for name in dotted.split('.'):
            if name.isdigit():
                mod = mod._modules[name]
            else:
                mod = getattr(mod, name)
        return mod

    @staticmethod
    def _auto_normalize_ckpt_state_dict(ckpt_sd: dict, target_keys: set) -> tuple[dict, str, int]:
        candidates = ["", "p_frame_model.", "model.", "module.", "net.", "video.", "video_model.", "p_model."]
        roots = {k.split('.', 1)[0] + "." for k in ckpt_sd.keys() if "." in k}
        candidates += sorted(roots)

        best_sd, best_pref, best_hits = {}, "", -1
        for pref in candidates:
            remap = {k[len(pref):]: v for k, v in ckpt_sd.items() if k.startswith(pref)}
            hits = sum(1 for k in remap.keys() if k in target_keys)
            if hits > best_hits:
                best_sd, best_pref, best_hits = remap, pref, hits
        return best_sd, best_pref, best_hits

    @staticmethod
    @torch.no_grad()
    def _inflate_input_convs_from_ckpt(model: nn.Module, ckpt_sd: dict,
                                       init_mode: str = "kaiming",
                                       zero_last: bool = False) -> list[str]:
        inflated = []
        model_sd = model.state_dict()
        for name, new_w in model_sd.items():
            if not (name.endswith(".weight") and isinstance(new_w, torch.Tensor) and new_w.ndim == 4):
                continue
            if name not in ckpt_sd:
                continue
            old_w = ckpt_sd[name]
            if not (isinstance(old_w, torch.Tensor) and old_w.ndim == 4):
                continue

            Cout_n, Cin_n, kH_n, kW_n = new_w.shape
            Cout_o, Cin_o, kH_o, kW_o = old_w.shape
            if (Cout_n == Cout_o) and (kH_n == kH_o) and (kW_n == kW_o) and (Cin_n == Cin_o + 1):
                stitched = torch.empty_like(new_w)
                stitched[:, :Cin_o, :, :] = old_w.to(dtype=new_w.dtype, device=new_w.device)

                extra = stitched[:, Cin_o:, :, :]
                if zero_last:
                    extra.zero_()
                elif init_mode == "kaiming":
                    nn.init.kaiming_normal_(extra, mode='fan_in', nonlinearity='leaky_relu')
                elif init_mode == "copy_mean":
                    extra.copy_(stitched[:, :Cin_o, :, :].mean(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_uniform_(extra, a=math.sqrt(5))

                mod_path = name.rsplit(".weight", 1)[0]
                conv = VideoCompressionTrainer._resolve_module(model, mod_path)  # <— use class/static reference
                conv.weight.data.copy_(stitched)
                inflated.append(name)
        return inflated

    def load_p_from_lightning_ckpt(self, strict: bool = False, map_location="cpu",
                                   inflate_init="kaiming", zero_last=False):
        path = self.config.video_checkpoint_path
        raw = torch.load(path, map_location=map_location)
        sd = raw.get("state_dict", raw)

        model_keys = set(self.p_frame_model.state_dict().keys())
        ckpt_sd, chosen_pref, hits = self._auto_normalize_ckpt_state_dict(sd, model_keys)  # <— self.
        if self.global_rank == 0:
            print(f"[CKPT] Using prefix '{chosen_pref}' with {hits} overlapping keys for P-frame model.")

        missing = self.p_frame_model.load_state_dict(ckpt_sd, strict=False)
        if self.global_rank == 0:
            print("[CKPT] Loaded P-frame (partial). Missing:", missing.missing_keys[:5], "...")
            print("[CKPT] Unexpected:", missing.unexpected_keys[:5], "...")

        inflated = self._inflate_input_convs_from_ckpt(self.p_frame_model, ckpt_sd,       # <— self.
                                                       init_mode=inflate_init,
                                                       zero_last=zero_last)
        if self.global_rank == 0:
            if inflated:
                print(f"[CKPT] Inflated input convs (+1 channel): {len(inflated)}")
                for n in inflated[:6]:
                    print("  -", n)
            else:
                print("[CKPT] No conv inflation needed.")

        print("Loaded P-frame weights from:", path)
        return missing

    def _load_checkpoints(self):
        if not self.config.image_checkpoint_path and not self.config.video_checkpoint_path:
            return

        assert self.config.image_checkpoint_path and os.path.exists(self.config.image_checkpoint_path), \
            f"Image model checkpoint not found: {self.config.image_checkpoint_path}"

        if self.global_rank == 0:
            print(f"Loading checkpoint for image model from {self.config.image_checkpoint_path}")

        self.i_frame_model.load_state_dict(
            get_state_dict(self.config.image_checkpoint_path, state_dict="model")
        )

        if self.config.video_checkpoint_path and os.path.exists(self.config.video_checkpoint_path):
            if self.global_rank == 0:
                print(f"Loading checkpoint for video model from {self.config.video_checkpoint_path}")
            # robust partial load + 3->4 inflation
            self.load_p_from_lightning_ckpt(strict=False, map_location="cpu", inflate_init="kaiming", zero_last=False)
        else:
            if self.global_rank == 0:
                print("No checkpoint for video model found, starting fresh.")

            

    @rank_zero_only
    def _setup_csv_logging(self):
        if self.logger is not None and hasattr(self.logger, 'log_dir'):
            experiment_log_dir = self.logger.log_dir
            if experiment_log_dir:
                self.csv_log_dir = os.path.join(experiment_log_dir, "csv_metrics")
                self.train_csv_path = os.path.join(self.csv_log_dir, "train_metrics.csv")
                self.val_csv_path = os.path.join(self.csv_log_dir, "val_metrics.csv")

                try:
                    os.makedirs(self.csv_log_dir, exist_ok=True)

                    train_headers = ["epoch", "step", "loss", "bpp", "bpp_y", "bpp_z", "psnr", "mse", "qp_avg"]
                    if not os.path.exists(self.train_csv_path):
                        with open(self.train_csv_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(train_headers)
                        self.train_headers_written = True
                        print(f"Train CSV log file created at: {self.train_csv_path}")

                    val_headers = ["epoch", "step", "loss", "bpp", "bpp_y", "bpp_z", "psnr", "mse"]
                    if not os.path.exists(self.val_csv_path):
                        with open(self.val_csv_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(val_headers)
                        self.val_headers_written = True
                        print(f"Validation CSV log file created at: {self.val_csv_path}")

                except Exception as e:
                    print(f"Error setting up CSV logging: {e}")
                    self.train_csv_path = None
                    self.val_csv_path = None

    def compute_psnr(self, mse: float, max_val: float = 1.0) -> float:
        mse_tensor = torch.tensor(mse)
        psnr = 10 * torch.log10(max_val ** 2 / (mse_tensor + 1e-12))
        return psnr.item()

    def compute_lambda(self, qp: int, device: torch.device):
        val = math.exp(math.log(self.lambda_min) +
                       qp / (self.q_levels - 1) * (math.log(self.lambda_max) - math.log(self.lambda_min)))
        return torch.tensor(val, device=device, dtype=torch.float32)

    """
    def rate_distortion_loss(self, results, target, qp: int, fa_idx: int, eval_mode: bool = False):
        w_t = 1.0 if eval_mode else self.weights_map[fa_idx]
        bpp   = results["bpp"].mean()
        bpp_y = results["bpp_y"].mean()
        bpp_z = results["bpp_z"].mean()
        mse   = F.mse_loss(results["dpb"]["frame"], target, reduction="mean")
        lam   = self.compute_lambda(qp, target.device)
        loss  = bpp_y + bpp_z + w_t * lam * mse
        return loss, bpp, bpp_y, bpp_z, mse
    """
    def rate_distortion_loss(self, results, target, qp: int, fa_idx: int,
                         eval_mode: bool = False, mask: torch.Tensor | None = None):
        """
        results['dpb']['frame'] and target are (B,3,H,W) in YCbCr.
        mask is (B,1,H,W) in {0,1}. If None, falls back to standard MSE.
        """
        w_t  = 1.0 if eval_mode else self.weights_map[fa_idx]
        bpp  = results["bpp"].mean()
        bpp_y = results["bpp_y"].mean()
        bpp_z = results["bpp_z"].mean()

        pred = results["dpb"]["frame"]

        if mask is None:
            mse = F.mse_loss(pred, target, reduction="mean")
            prev_obj = mse
        else:
            m = (mask > 0).to(pred.dtype).expand_as(pred)  # 1 inside, 0 outside
            if m.sum() == 0:
                # No masked pixels -> standard MSE
                mse = F.mse_loss(pred, target, reduction="mean")
                prev_obj = mse
            else:
                # Masked = 100x, non-masked = 1x
                w = 1.0 + 100.0 * m
                mse = F.mse_loss(pred, target, reduction="mean", weight=w)
                prev_obj = F.mse_loss(pred, target, reduction="mean")

        lam  = self.compute_lambda(qp, target.device)
        loss = bpp_y + bpp_z + w_t * lam * mse
        return loss, bpp, bpp_y, bpp_z, mse, prev_obj
        
    """
    def compute_wmse(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None):
        # per-pixel SE averaged over channels -> (B,1,H,W)
        se = (recon - target) ** 2
        se = se.mean(dim=1, keepdim=True)
        if mask is None:
            return se.mean()
        weights = 1.0 + (self.mask_fg_weight - 1.0) * mask
        return (weights * se).sum() / weights.sum().clamp_min(1e-6)
        
    @torch.no_grad()
    def masked_percentile_mse(self, recon: torch.Tensor, target: torch.Tensor,
                            mask: torch.Tensor | None, p: torch.Tensor) -> torch.Tensor:
    """
    """
        Returns the per-batch average of the p-quantile of per-pixel MSE over the ROI (mask==1).
        Mathematically: phi = mean_b Q_p( {( (recon-target)^2 averaged over channels ) on ROI_b } )
        If ROI is empty for a sample, falls back to all pixels for that sample.
    """
    """
        err = ((recon - target) ** 2).mean(dim=1, keepdim=True)
        B = err.size(0)
        out = []
        p_val = float(p.clamp(0.0, 1.0).item())
        for b in range(B):
            e = err[b].flatten()  # (H*W,)
            if mask is not None:
                m = (mask[b].flatten() > 0.5)
                if m.any():
                    e = e[m]
            q = torch.quantile(e, q=p_val)
            out.append(q)
        return torch.stack(out).mean() 

    def compute_wmse(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        recon_f32 = recon.float()
        target_f32 = target.float()
        if mask is None:
            return torch.mean((recon_f32 - target_f32) ** 2)
        se = (recon_f32 - target_f32) ** 2               # (B,3,H,W)
        se = se.mean(dim=1, keepdim=True)                # (B,1,H,W)
        weights = (self.mask_fg_weight - 1.0) * mask.float()
        return (weights * se).sum() / weights.sum().clamp_min(1e-6)
    """
    def unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def training_step(self, batch, batch_idx):
        if hasattr(self, "p_frame_model"):
            self.debug.register_conv_hooks(self.p_frame_model, limit=8)

        self.i_frame_model.eval()
        self.p_frame_model.train()

        # 3 optimizers: backbone (main), probe, aux(bit_estimator)
        opt_main, opt_probe, opt_aux = self.optimizers()
        acc_steps = self.trainer.accumulate_grad_batches

        opt_main.zero_grad(set_to_none=True)
        opt_probe.zero_grad(set_to_none=True)
        opt_aux.zero_grad(set_to_none=True)

        _, frames, masks = batch
        frames = frames.to(self.device, non_blocking=True)
        masks  = masks.to(self.device,  non_blocking=True)  # (B,T,1,H,W)
        bs, seq_len = frames.shape[:2]
        qp = random.randint(0, 63)

        # cosine schedule
        total_iter = self.trainer.max_epochs * len(self.trainer.train_dataloader)
        lr_now     = self.get_lr(self._batches_seen, total_iter)

        # backbone: slow lr
        opt_main.param_groups[0]["lr"] = lr_now * 0.3
        # probe: full lr
        opt_probe.param_groups[0]["lr"] = lr_now
        # aux: fixed aux lr
        opt_aux.param_groups[0]["lr"]  = self.config.optimizer.aux_lr

        self._batches_seen += 1


        # metrics accumulators
        loss_sum = bpp_sum = bpp_y_sum = bpp_z_sum = mse_sum = 0.0
        frame_cnt = micro_step = 0
        qp_sum = qp

        #with torch.no_grad():
        #    dpb = self.i_frame_model(frames[:, 0], qp)["dpb"]

        tau_wmse = 0

        with torch.no_grad():
            i_out = self.i_frame_model(frames[:, 0], qp)
            dpb   = i_out["dpb"]

        for t in range(1, seq_len):
            fa_idx = index_map[t % 8]
            curr_qp = self.unwrap(self.p_frame_model).shift_qp(qp, fa_idx)

            if MASK_TRAIN:
                if t == 1 or current_mask is None:
                    current_mask = masks[:, t]
            else:
                current_mask = masks[:, t]

            x_in = torch.cat([frames[:, t], current_mask], dim=1)
            results = self.p_frame_model(x_in, curr_qp, dpb, after_i=(t == 1))
            

            if not CONSTRAINT_OPT:
                loss, bpp, bpp_y, bpp_z, mse, prev_obj = self.rate_distortion_loss(
                results, frames[:, t], qp, fa_idx, mask=masks[:, t])

            else:
                results = self.p_frame_model(x_in, curr_qp, dpb, after_i=(t == 1))
                rd_loss, bpp, bpp_y, bpp_z, mse, prev_obj = self.rate_distortion_loss(
                    results, frames[:, t], qp, fa_idx, mask=None
                )
                # Base objective: minimize rate R only
                R = (bpp_y + bpp_z)
                # get effective QP
                if isinstance(curr_qp, (int, float)):
                    qp_eff = int(round(curr_qp))
                elif torch.is_tensor(curr_qp):
                    qp_eff = int(round(float(curr_qp.detach().cpu())))
                else:
                    qp_eff = int(qp)

                psnrm_target_db = self._psnrm_target_for_qp(qp_eff)
                tau_qp = self._mse_from_psnr_db(psnrm_target_db)
                roi_mse = self._roi_mse(results["dpb"]["frame"], frames[:, t], masks[:, t])
                g = (roi_mse - roi_mse.new_tensor(tau_qp)) / (roi_mse.new_tensor(tau_qp) + 1e-12)  # normalized

                alm_term = self._alm_term_from_g(g)
                used_loss = R + float(self.config.alm_penalty_scale)* alm_term

                # accumulate violation for μ update
                """
                self.alm_h_accum.add_(g.detach())
                self.alm_h_count.add_(1.0)
                """
                loss = used_loss

            if MASK_TRAIN:
                # Just predict the mask and compare with GT
                pred_mask = results.get("mask_pred", None)
                if pred_mask is None:
                    #print("we was here")
                    dpb = {k: (v.detach() if isinstance(v, torch.Tensor) else v)
                   for k, v in results["dpb"].items()}
                    continue
                
                print("and hopefully here")
                gt_mask = masks[:, t]  # shape: (B,1,H,W)

                # BCE loss is common for binary masks; use Dice if you'd prefer
                bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)

                loss =+ bce
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] Epoch {self.current_epoch}/{self.trainer.max_epochs}, "
                  f"Step {batch_idx}, Loss {bce:.6f}", flush=True)
            
                


            self.manual_backward(loss / acc_steps)
            micro_step += 1

            # accumulate
            frame_cnt += 1
            loss_sum  += loss.item()
            bpp_sum   += bpp.item()
            bpp_y_sum += bpp_y.item()
            bpp_z_sum += bpp_z.item()
            mse_sum   += mse.item()

            # optimizer step on accumulation boundary
                        # optimizer step on accumulation boundary
            last_frame = (t == seq_len - 1)
            if micro_step % acc_steps == 0 or last_frame:
                scaler = getattr(self.trainer.precision_plugin, "scaler", None)
                if scaler is not None:
                    scaler.unscale_(opt_main)
                    scaler.unscale_(opt_probe)
                    scaler.unscale_(opt_aux)

                torch.nn.utils.clip_grad_norm_(self.p_frame_model.parameters(), self.config.grad_clip)

                try:
                    gnorm = self.debug.model_grad_norm(self.p_frame_model)
                    self.debug._log(f"GRAD_NORM before step: {gnorm:.3e}")
                except Exception:
                    pass

                # step all three optimizers
                opt_main.step()
                opt_probe.step()
                opt_aux.step()

                if CONSTRAINT_OPT:
                    self._alm_dual_update()

                try:
                    pnorm = self.debug.model_param_norm(self.p_frame_model)
                    self.debug._log(f"[DBG] PARAM_NORM after step: {pnorm:.3e}")
                except Exception:
                    pass

                opt_main.zero_grad(set_to_none=True)
                opt_probe.zero_grad(set_to_none=True)
                opt_aux.zero_grad(set_to_none=True)


                if CONSTRAINT_OPT:
                    self._alm_dual_update()


                try:
                    pnorm = self.debug.model_param_norm(self.p_frame_model)
                    self.debug._log(f"[DBG] PARAM_NORM after step: {pnorm:.3e}")
                except Exception:
                    pass
                
                opt_main.zero_grad(set_to_none=True)
                opt_aux.zero_grad(set_to_none=True)
                opt_probe.zero_grad(set_to_none=True)

            dpb = {k: (v.detach() if isinstance(v, torch.Tensor) else v)
                   for k, v in results["dpb"].items()}

        avg_loss  = loss_sum  / frame_cnt
        avg_bpp   = bpp_sum   / frame_cnt
        avg_bpp_y = bpp_y_sum / frame_cnt
        avg_bpp_z = bpp_z_sum / frame_cnt
        avg_mse   = mse_sum   / frame_cnt
        psnr      = self.compute_psnr(avg_mse)

        # logs
        self.log("train/loss",  avg_loss,  on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/bpp",   avg_bpp,   on_step=True, sync_dist=True)
        self.log("train/bpp_y", avg_bpp_y, on_step=True, sync_dist=True)
        self.log("train/bpp_z", avg_bpp_z, on_step=True, sync_dist=True)
        self.log("train/psnr",  psnr,      on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/mse",   avg_mse,   on_step=True, sync_dist=True)
        self.log("train/qp",    float(qp), on_step=True, sync_dist=True)
        self.log("train/lr_main", lr_now,  on_step=True, sync_dist=True)
        self.log("train/lr_aux", self.config.optimizer.aux_lr, on_step=True, sync_dist=True)

        if batch_idx % self.config.log_interval == 0:
            self._log_train_metrics_to_csv({
                "epoch": self.current_epoch,
                "step":  self.global_step,
                "loss":  avg_loss,
                "bpp":   avg_bpp,
                "bpp_y": avg_bpp_y,
                "bpp_z": avg_bpp_z,
                "psnr":  psnr,
                "mse":   avg_mse,
                "qp_avg": float(qp_sum / self._batches_seen),
                "tau_wmse": tau_wmse
            })
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] Epoch {self.current_epoch}/{self.trainer.max_epochs}, "
                  f"Step {batch_idx}, Loss {avg_loss:.6f}, PSNR {psnr:.2f}, "
                  f"BPP {avg_bpp:.5f}, MSE/WMSE {avg_mse:.5f}"
                  f"Prev_Obj {prev_obj:.5f}", flush=True)
            self._log_images(frames[0, 0], results["dpb"]["frame"][0], "train")

        return torch.tensor(avg_loss, device=self.device)

    def validation_step(self, batch, batch_idx):
        self.i_frame_model.eval()
        self.p_frame_model.eval()

        if batch_idx % 10 == 0:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Validation batch {batch_idx}")

        _, rgb_data, masks = batch
        batch = rgb_data.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)
        bs, seq_len = batch.size(0), batch.size(1)
        frames = [batch[:, i, ...] for i in range(seq_len)]
        masks_t = [masks[:, i, ...] for i in range(seq_len)]
        qp = random.randint(0, 63)

        total_loss = 0
        total_bpp = 0
        total_bpp_y = 0
        total_bpp_z = 0
        total_mse = 0
        frame_count = 0

        with torch.no_grad():
            dpb = self.i_frame_model(frames[0], qp)["dpb"]

            for frame_idx in range(1, seq_len):
                fa_idx = index_map[frame_idx % 8]
                curr_qp = self.unwrap(self.p_frame_model).shift_qp(qp, fa_idx)

                results = self.p_frame_model(frames[frame_idx], curr_qp, dpb, after_i=(frame_idx == 1))
                if CONSTRAINT_OPT:
                    loss, bpp, bpp_y, bpp_z, mse, prev_obj = self.rate_distortion_loss(
                results, frames[frame_idx], qp, fa_idx, eval_mode=True, mask=masks_t[frame_idx])
                else:
                    loss, bpp, bpp_y, bpp_z, mse, prev_obj = self.rate_distortion_loss(
                results, frames[frame_idx], qp, fa_idx, eval_mode=True, mask=masks_t[frame_idx])
                dpb = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in results["dpb"].items()}

                total_loss += loss
                total_bpp += bpp.item()
                total_bpp_y += bpp_y.item()
                total_bpp_z += bpp_z.item()
                total_mse += mse.item()
                frame_count += 1

        avg_loss = total_loss.item() / frame_count
        avg_bpp  = total_bpp / frame_count
        avg_bpp_y = total_bpp_y / frame_count
        avg_bpp_z = total_bpp_z / frame_count
        avg_mse  = total_mse / frame_count
        psnr     = self.compute_psnr(avg_mse)

        if batch_idx % 5 == 0:
            print(f"Val batch {batch_idx}: Loss={avg_loss:.6f}, PSNR={psnr:.2f}, BPP={avg_bpp:.6f}")

        self.log("val/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/bpp",  avg_bpp,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/bpp_y", avg_bpp_y, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/bpp_z", avg_bpp_z, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mse",  avg_mse, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self._log_images(frames[0][0], results["dpb"]["frame"][0], "val")

        return {"val_loss": avg_loss, "val_bpp": avg_bpp, "val_psnr": psnr, "val_mse": avg_mse}
    
    @rank_zero_only
    def on_validation_epoch_start(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Starting validation for epoch {self.current_epoch}")
    def on_train_epoch_start(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Starting training epoch {self.current_epoch}/{self.trainer.max_epochs}")

    def on_train_epoch_end(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Training epoch {self.current_epoch} completed")

    @rank_zero_only
    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics if self.trainer else {}
        default_tensor = torch.tensor(0.0, device=self.device)

        val_loss = metrics.get("val/loss", default_tensor).item()
    
        val_psnr = metrics.get("val/psnr", default_tensor).item()
        val_bpp  = metrics.get("val/bpp",  default_tensor).item()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}]   epoch {self.current_epoch} completed:")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val PSNR: {val_psnr:.2f} dB")
        print(f"  Val BPP:  {val_bpp:.6f}")
        print("-" * 50)

        log_data = {
            "epoch": self.current_epoch,
            "step": self.global_step,
            "loss": val_loss,
            "bpp": val_bpp,
            "bpp_y": metrics.get("val/bpp_y", default_tensor).item(),
            "bpp_z": metrics.get("val/bpp_z", default_tensor).item(),
            "psnr": val_psnr,
            "mse": metrics.get("val/mse", default_tensor).item(),
        }
        self._log_val_metrics_to_csv(log_data)

    def get_lr(self, it: int, total_iter: int):
        if self.config.optimizer.warmup_iters > 0 and it < self.config.optimizer.warmup_iters:
            return self.config.optimizer.base_lr * it / self.config.optimizer.warmup_iters
        decay_ratio = (it - self.config.optimizer.warmup_iters) / max(1, total_iter - self.config.optimizer.warmup_iters)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.optimizer.min_lr + coeff * (self.config.optimizer.base_lr - self.config.optimizer.min_lr)

    def configure_optimizers(self):
        backbone_params, probe_params, aux_params = self._split_probe_backbone()

        opt_type = self.config.optimizer.optimizer_type.lower()
        base_lr  = self.config.optimizer.base_lr
        aux_lr   = self.config.optimizer.aux_lr
        wd       = self.config.optimizer.weight_decay

        def make_opt(params, lr, weight_decay):
            if opt_type == "adamw":
                return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            elif opt_type == "adam":
                return optim.Adam(params, lr=lr, weight_decay=weight_decay)
            else:
                opt_cls = Lion
                if opt_cls is None:
                    raise RuntimeError("Lion optimizer requested but timm's Lion is not available.")
                return opt_cls(params, lr=lr, weight_decay=weight_decay)

        opt_main  = make_opt(backbone_params, lr=base_lr * 0.3, weight_decay=wd * 0.5)
        # probe: full LR + strong WD
        opt_probe = make_opt(probe_params,    lr=base_lr,       weight_decay=wd)
        # aux: higher lr, no wd
        opt_aux   = make_opt(aux_params,     lr=aux_lr,         weight_decay=wd)

        # We now have 3 optimizers: backbone, probe, aux(bit_estimator)
        return [opt_main, opt_probe, opt_aux]

    def _log_images(self, original, recon, prefix):
        if self.global_rank != 0:
            return
        try:
            original_rgb = ycbcr2rgb(original.unsqueeze(0)).squeeze(0)
            recon_rgb    = ycbcr2rgb(recon.unsqueeze(0)).squeeze(0)
            original_prep = self._prepare_image_for_logging(original_rgb)
            recon_prep    = self._prepare_image_for_logging(recon_rgb)
            if self.logger and hasattr(self.logger, 'experiment'):
                if isinstance(self.logger, TensorBoardLogger):
                    tb_writer = self.logger.experiment
                    if original_prep is not None:
                        tb_writer.add_image(f"{prefix}/Original", original_prep, self.global_step)
                    if recon_prep is not None:
                        tb_writer.add_image(f"{prefix}/Reconstructed", recon_prep, self.global_step)
        except Exception as e:
            print(f"Error during image logging: {e}")

    def _prepare_image_for_logging(self, img_tensor):
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

    @rank_zero_only
    def _log_metrics_to_csv(self, file_path, headers, metrics_dict, headers_written_flag):
        if not file_path:
            return
        try:
            file_exists = os.path.exists(file_path)
            headers_written = getattr(self, headers_written_flag)
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists or not headers_written:
                    writer.writerow(headers)
                    setattr(self, headers_written_flag, True)
                row_values = [metrics_dict.get(h, "") for h in headers]
                writer.writerow(row_values)
        except Exception as e:
            print(f"Error writing to CSV log {os.path.basename(file_path)}: {e}")

    @rank_zero_only
    def _log_train_metrics_to_csv(self, metrics_dict):
        headers = ["epoch", "step", "loss", "bpp", "bpp_y", "bpp_z", "psnr", "mse", "qp_avg"]
        self._log_metrics_to_csv(self.train_csv_path, headers, metrics_dict, "train_headers_written")

    @rank_zero_only
    def _log_val_metrics_to_csv(self, metrics_dict):
        headers = ["epoch", "step", "loss", "bpp", "bpp_y", "bpp_z", "psnr", "mse"]
        self._log_metrics_to_csv(self.val_csv_path, headers, metrics_dict, "val_headers_written")

# ------------------------------ Entrypoint ------------------------------

def main(config: DictConfig):
    pl.seed_everything(config.seed)

    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    # Build seg-waymo DataModule; we pass glob for TFRecords
    tf_glob = os.path.join(config.dataset.data_dir, "*.tfrecord")
    # pick crop size from either crop_size or crop[0]
    if config.dataset.crop_size is not None:
        crop_size = config.dataset.crop_size
    else:
        c = config.dataset.crop
        crop_size = int(c[0] if isinstance(c, (list, tuple)) else c)

    if config.build_cache:
        build_cache(
        tf_glob=tf_glob,
        cache_dir="seg_cache",
        yolo_weights="yolov8x-seg.pt",   # segmentation weights
        storage_format="npz",                      # or "png"
        classes_keep=[0, 2, 5], 
        thr=0.5,
        min_area=64,
        morph="open", morph_ksize=3,
        device="cuda"
        )
    datamodule = WaymoDataModule(
        tf_glob=tf_glob,
        seg_cache_dir="seg_cache",
        seq_len=(config.dataset.seq_len or config.dataset.n_frames),
        slide=config.dataset.slide,
        crop_size=crop_size,
        train_val_test=config.dataset.train_val_test_split,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        strict_masks=True,
        seed=config.seed,
    )

    start_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.exp_name}_{start_time_str}"

    tb_logger = TensorBoardLogger(save_dir=config.log_dir, name=experiment_name, version="")
    log_dir = tb_logger.log_dir
    print(f"Logging to directory: {log_dir}")

    try:
        os.makedirs(log_dir, exist_ok=True)
        config_save_path = os.path.join(log_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            OmegaConf.save(config=config, f=f)
        print(f"Saved configuration to: {config_save_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")

    model = VideoCompressionTrainer(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="{epoch:02d}-{val/loss:.6f}",
        save_top_k=config.save_top_k,
        mode="min",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    summary = ModelSummary(max_depth=3)

    callbacks_list = [checkpoint_callback, lr_monitor, summary]
    if config.num_gpus > 0:
        callbacks_list.append(DeviceStatsMonitor())

    strategy = "auto"
    if config.num_gpus > 1:
        strategy = "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if config.num_gpus > 0 else "cpu",
        devices=config.num_gpus if config.num_gpus > 0 else "auto",
        strategy=strategy,
        logger=tb_logger,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        precision=config.precision,
        callbacks=callbacks_list,
        enable_progress_bar=False,
    )

    print("Starting training...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=config.resume_from_checkpoint,
    )
    print("Training finished.")

if __name__ == "__main__":
    default_config_path = "video_compression_config.yaml"

    if not os.path.exists(default_config_path):
        print(f"Creating default config file at: {default_config_path}")
        default_cfg_obj = OmegaConf.structured(VideoCompressionConfig)
        try:
            with open(default_config_path, "w") as f:
                OmegaConf.save(config=default_cfg_obj, f=f)
            print("Default config file created. Please review and edit as needed.")
        except Exception as e:
            print(f"Error creating default config file: {e}")

    print(f"Loading configuration from {default_config_path}...")
    try:
        conf = OmegaConf.load(default_config_path)
        cli_conf = OmegaConf.from_cli()
        conf = OmegaConf.merge(conf, cli_conf)
        conf = OmegaConf.merge(OmegaConf.structured(VideoCompressionConfig), conf)

        print("Final configuration:")
        print(OmegaConf.to_yaml(conf))

    except FileNotFoundError:
        print(f"Error: Configuration file '{default_config_path}' not found.")
        raise SystemExit
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise SystemExit

    main(conf)

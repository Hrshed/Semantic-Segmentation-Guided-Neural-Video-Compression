import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Union
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def is_main_process():
    try:
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True

def visualize_q_scale(
    q_scale_value: "torch.Tensor",
    save_dir: Union[str, os.PathLike],
    *,
    q_scale_name: str = "q_scale_enc",
    epoch: int | None = None,
    log_to_wandb: bool = False,
    ) -> tuple[str, str]:
    """Create an animated histogram (GIF) and a mean/std curve (PNG).

    Parameters
    ----------
    q_scale_value
        Tensor whose first dimension is the QP index.
    save_dir
        Output directory (will be created if necessary).
    q_scale_name
        Prefix for generated filenames.
    epoch
        Optional epoch index added to filenames.
    log_to_wandb
        Log the generated images to Weights & Biases when *True*.

    Returns
    -------
    gif_path, png_path
        Paths of the saved files.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy
    param_all = q_scale_value.detach().cpu().numpy().squeeze()

    # ========== Animated Histogram ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(param_all.min(), param_all.max(), 100)

    def update(frame):
        ax.clear()
        param = param_all[frame]
        mean = param.mean()
        std = param.std()
        ax.hist(param, bins=bins, alpha=0.7)
        ax.set_title(f"{q_scale_name}[{frame}]  mean={mean:.4f}, std={std:.4f}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True)

    gif_name = f"{q_scale_name}_epoch_{epoch}.gif"
    gif_path = os.path.join(save_dir, gif_name)

    ani = animation.FuncAnimation(fig, update, frames=param_all.shape[0], interval=300)
    ani.save(gif_path, writer="pillow", fps=10)
    plt.close(fig)

    # ========== Mean/Std Curve ==========
    means = param_all.mean(axis=1)
    stds = param_all.std(axis=1)
    qps = np.arange(param_all.shape[0])

    plot_name = f"{q_scale_name}_epoch_{epoch}.png"
    plot_path = os.path.join(save_dir, plot_name)

    plt.figure(figsize=(10, 6))
    plt.plot(qps, means, label="Mean", marker='o')
    plt.plot(qps, stds, label="Std", marker='s')
    plt.xlabel("QP Index")
    plt.ylabel("Value")
    plt.title(f"Mean and Std of {q_scale_name} vs QP")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # ========== Log to WandB ==========
    if log_to_wandb and WANDB_AVAILABLE and is_main_process():
        wandb.log({
            f"{q_scale_name}/mean_std": wandb.Image(plot_path),
            f"{q_scale_name}/histogram": wandb.Video(gif_path, format="gif"),
        }, step=epoch)

    return gif_path, plot_path


import random
import cv2
from simple_waymo_open_dataset_reader import WaymoDataFileReader, utils

def project_top_lidar(
    file_path: str,
    camera_name: int,
    lidar_name: int, 
    seq_len: int = 1,
    start_idx: int = 0,
    H = 1280, 
    W = 1920):

    """Project TOP LiDAR points onto a camera image plane.

    The output image contains depth, intensity and elongation in three
    channels. Pixels without a LiDAR hit are zero.

    Returns
    -------
    proj
        (3, H, W) float32 array.
    frame
        The chosen Waymo frame object.
    front_pts
        LiDAR points (vehicle frame) inside the camera FOV.
    rgb
        Front RGB image
    """

    def _rgb_from_proto(proto) -> torch.Tensor:
        img_bgr = cv2.imdecode(np.frombuffer(proto.image, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0

    frames = list(WaymoDataFileReader(file_path))

    if start_idx < 0 or start_idx + seq_len > len(frames):
        raise ValueError(
            f"Invalid start_idx ({start_idx}) or seq_len ({seq_len}) for total {len(frames)} frames.")
    
    chosen_frames = frames[start_idx : start_idx + seq_len]

    samples = []
    for frame in chosen_frames:
        # Camera image ------------------------------------------------------
        img_proto = next(i for i in frame.images if i.name == camera_name)
        rgb = _rgb_from_proto(img_proto)

        # LiDAR → point cloud ---------------------------------------------
        laser = next(l for l in frame.lasers if l.name == lidar_name)
        ri, cam_proj, _ = utils.parse_range_image_and_camera_projection(laser)
        pcl, _ = utils.project_to_pointcloud(
            frame, ri, cam_proj, _, utils.get(frame.context.laser_calibrations, lidar_name)
        )

        valid = ri[..., 0] > 0
        pcl = pcl.reshape(-1, 3)
        cam_proj = cam_proj.reshape(-1, 6)[valid.reshape(-1)]

        cam_id, u_all, v_all, *_ = cam_proj.T
        in_view = (
            (cam_id == camera_name)
            & (u_all >= 0)
            & (u_all < W)
            & (v_all >= 0)
            & (v_all < H))

        u_px = u_all[in_view].astype(np.int32)
        v_px = v_all[in_view].astype(np.int32)
        front_pts = pcl[in_view]

        # Depth
        cam_cal = next(c for c in frame.context.camera_calibrations if c.name == camera_name)
        T_c2v = np.asarray(cam_cal.extrinsic.transform, dtype=np.float32).reshape(4, 4)
        T_v2c = np.linalg.inv(T_c2v)
        pts_cam = (T_v2c @ np.c_[front_pts, np.ones(len(front_pts))].T).T
        depth_cam = pts_cam[:, 0]

        # Intensity & elongation 
        intensity = ri[..., 1][valid][in_view].astype(np.float32)
        elong = ri[..., 2][valid][in_view].astype(np.float32)

        # Build projection map 
        proj = np.full((3, H, W), fill_value=0, dtype=np.float32)
        proj[0, v_px, u_px] = depth_cam / 75.0
        proj[1, v_px, u_px] = np.clip(intensity, 0, 1.5) / 1.5
        proj[2, v_px, u_px] = elong / 1.5

        samples.append((proj, frame, front_pts, rgb))

    return samples

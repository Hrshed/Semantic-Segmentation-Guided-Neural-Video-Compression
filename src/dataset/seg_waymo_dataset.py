# waymo_dataset.py
# Read-only Waymo dataset that returns YCbCr + binary mask from cache.
# Output per __getitem__: (proj_seq, ycbcrm_seq)
#   proj_seq   : (S, 3, H, W)   (simple lidar projection features)
#   ycbcrm_seq : (S, 4, H, W)   [Y, Cb, Cr, Mask] in [0,1], Mask ∈ {0,1}

from __future__ import annotations

import glob
import itertools
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, utils


# ----------------------------- Color helpers (BT.709) -----------------------------

def _rgb_from_proto(img_proto) -> torch.Tensor:
    """Decode Waymo image proto -> RGB tensor (3,H,W) in [0,1]."""
    arr = np.frombuffer(img_proto.image, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed for an image in the TFRecord.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.as_tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0


def _rgb_to_ycbcr_bt709(rgb_chw: torch.Tensor) -> torch.Tensor:
    """RGB(3,H,W)[0..1] -> YCbCr(3,H,W)[0..1], BT.709 constants."""
    r, g, b = rgb_chw[0], rgb_chw[1], rgb_chw[2]
    Kr, Kg, Kb = 0.2126, 0.7152, 0.0722
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    return torch.stack([y, cb, cr], dim=0).clamp(0.0, 1.0)


# ------------------------------- Mask cache I/O ----------------------------------

def _mask_paths(cache_dir: str | Path, tf_path: str, frame_idx: int) -> Tuple[Path, Path]:
    """Return (<frame>.npz, <frame>.png) candidate paths."""
    base = Path(tf_path).stem
    d = Path(cache_dir) / base
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{frame_idx:06d}.npz", d / f"{frame_idx:06d}.png"


def _load_cached_mask(cache_dir: str | Path, tf_path: str, frame_idx: int, H: int, W: int) -> torch.Tensor:
    """
    Load cached mask as (1,H,W) float tensor in {0,1}. Tries .npz then .png.
    Raises if missing or shape mismatch (so problems surface early).
    """
    p_npz, p_png = _mask_paths(cache_dir, tf_path, frame_idx)

    if p_npz.exists():
        data = np.load(p_npz, allow_pickle=False)
        if "mask" not in data:
            raise FileNotFoundError(f"NPZ found but no 'mask' array: {p_npz}")
        m = np.array(data["mask"], dtype=np.uint8)  # 0/1
        if m.shape != (H, W):
            raise ValueError(f"Mask size mismatch at {p_npz} (got {m.shape}, need {(H, W)})")
        return torch.from_numpy(m.astype(np.float32))[None, ...]

    if p_png.exists():
        m8 = cv2.imread(str(p_png), cv2.IMREAD_GRAYSCALE)
        if m8 is None:
            raise FileNotFoundError(f"Could not read PNG: {p_png}")
        if m8.shape != (H, W):
            raise ValueError(f"Mask size mismatch at {p_png} (got {m8.shape}, need {(H, W)})")
        m = (m8 > 127).astype(np.uint8)
        return torch.from_numpy(m.astype(np.float32))[None, ...]

    raise FileNotFoundError(f"Mask missing for frame {frame_idx}: {p_npz} OR {p_png}")


# ----------------------------- Simple lidar projection ---------------------------

def _project_top_lidar(frame, camera_name, lidar_name) -> torch.Tensor:
    """
    Builds a simple 3-channel projection tensor aligned to the camera image size.
    Channel 0 marks pixels where top-lidar points project into the camera frustum.
    Channels 1/2 are left as zeros (you can replace with intensity/elongation if desired).
    """
    img_proto = next(i for i in frame.images if i.name == camera_name)
    rgb = _rgb_from_proto(img_proto)
    _, H, W = rgb.shape

    laser = next(l for l in frame.lasers if l.name == lidar_name)
    ri, cam_proj, ri_pose = utils.parse_range_image_and_camera_projection(laser)

    # Project to 3D points in vehicle frame, then keep those that project to this camera
    pcl, _ = utils.project_to_pointcloud(
        frame,
        ri,
        cam_proj,
        ri_pose,
        utils.get(frame.context.laser_calibrations, lidar_name),
    )

    # Valid range pixels
    valid_mask = ri[..., 0] > 0
    cam_proj = cam_proj.reshape(-1, 6)[valid_mask.reshape(-1)]
    # cam_proj columns: [camera_id, u, v, ...] in pixel coordinates
    cam_id, u_px_all, v_px_all, *_ = cam_proj.T

    front_mask = (
        (cam_id == camera_name)
        & (u_px_all >= 0)
        & (u_px_all < W)
        & (v_px_all >= 0)
        & (v_px_all < H)
    )
    u_px = u_px_all[front_mask].astype(np.int32)
    v_px = v_px_all[front_mask].astype(np.int32)

    proj = np.zeros((3, H, W), dtype=np.float32)
    proj[0, v_px, u_px] = 1.0  # simple occupancy flag
    return torch.from_numpy(proj)


# --------------------------------- Dataset class ---------------------------------

class WaymoDataset(Dataset):
    """
    Read-only dataset: returns (proj_seq, ycbcrm_seq) where ycbcrm = [Y, Cb, Cr, Mask].
    Never runs YOLO; expects masks already cached as .npz (mask array) or .png.
    """

    def __init__(
        self,
        tfrecord_paths: Sequence[str] | str,
        seg_cache_dir: str = "seg_cache",
        seq_len: int = 8,
        slide: int = 1,
        crop_size: Optional[int] = 256,
        camera_name=dataset_pb2.CameraName.FRONT,
        lidar_name=dataset_pb2.LaserName.TOP,
        strict_masks: bool = True,
    ) -> None:
        super().__init__()

        # Expand TFRecord glob(s)
        if isinstance(tfrecord_paths, str):
            tfrecord_paths = sorted(glob.glob(tfrecord_paths))
        self.tfrecord_paths = list(tfrecord_paths)
        if not self.tfrecord_paths:
            raise ValueError("No TFRecord files found.")

        self.seg_cache_dir = seg_cache_dir
        self.seq_len = int(seq_len)
        self.slide = int(slide)
        self.crop_size = int(crop_size) if crop_size is not None else None
        self.camera_name = camera_name
        self.lidar_name = lidar_name
        self.strict_masks = strict_masks

        # Build (path, start) index covering all sequences
        self._index: List[Tuple[str, int]] = []
        for p in self.tfrecord_paths:
            total = sum(1 for _ in WaymoDataFileReader(p))
            if total >= self.seq_len:
                self._index.extend((p, s) for s in range(0, total - self.seq_len + 1, self.slide))

        if not self._index:
            raise ValueError("No index entries could be built (check seq_len/slide vs TFRecords).")

        # Reader cache (per-process)
        self._reader_cache: Dict[str, WaymoDataFileReader] = {}

    def __len__(self) -> int:
        return len(self._index)

    def _get_reader(self, path: str) -> WaymoDataFileReader:
        rdr = self._reader_cache.get(path)
        if rdr is None:
            rdr = WaymoDataFileReader(path)
            self._reader_cache[path] = rdr
        return rdr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tf_path, start = self._index[idx]
        #reader = self._get_reader(tf_path)
        reader = WaymoDataFileReader(tf_path)

        proj_seq: List[torch.Tensor] = []
        ycbcr_seq: List[torch.Tensor] = []
        mask_seq: List[torch.Tensor] = []

        #frames = itertools.islice(reader, start, start + self.seq_len)
        frames = itertools.islice(iter(reader), start, start + self.seq_len)

        proj_seq, ycbcr_seq, mask_seq = [], [], []
        for frame_idx, frame in zip(range(start, start + self.seq_len), frames):
            # RGB from proto
            img_proto = next(i for i in frame.images if i.name == self.camera_name)
            rgb = _rgb_from_proto(img_proto)  # (3,H,W) in [0,1]
            _, H, W = rgb.shape

            # Load cached mask (1,H,W) in {0,1}
            try:
                mask = _load_cached_mask(self.seg_cache_dir, tf_path, frame_idx, H, W)
            except FileNotFoundError as e:
                if self.strict_masks:
                    raise
                # fallback to zeros when strict_masks=False
                mask = torch.zeros(1, H, W, dtype=torch.float32)

            # Convert RGB -> YCbCr
            ycbcr = _rgb_to_ycbcr_bt709(rgb)  # (3,H,W)

            # Lidar projection (3,H,W)
            proj = _project_top_lidar(frame, self.camera_name, self.lidar_name)

            proj_seq.append(proj)
            ycbcr_seq.append(ycbcr)
            mask_seq.append(mask)

        if not ycbcr_seq:
            raise RuntimeError(
                f"No frames read for {Path(path).name} indices [{start}:{start+self.seq_len}). "
                "This usually happens if a cached reader iterator was reused and exhausted.")
        
        # Random crop (apply same crop to all tensors)
        if self.crop_size is not None:
            _, H, W = ycbcr_seq[0].shape
            ch, cw = int(self.crop_size), int(self.crop_size)
            if ch > H or cw > W:
                raise ValueError(f"crop_size {self.crop_size} exceeds image size {(H, W)}")
            top = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            proj_seq  = [p[:, top:top + ch, left:left + cw] for p in proj_seq]
            ycbcr_seq = [y[:, top:top + ch, left:left + cw] for y in ycbcr_seq]
            mask_seq  = [m[:, top:top + ch, left:left + cw] for m in mask_seq]

        # Concatenate mask as channel 4: [Y, Cb, Cr, Mask]
        ycbcrm_seq = [torch.cat([y, m], dim=0) for y, m in zip(ycbcr_seq, mask_seq)]

        return torch.stack(proj_seq, dim=0), torch.stack(ycbcrm_seq, dim=0)

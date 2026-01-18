from __future__ import annotations

import glob
import random
from typing import Dict, List, Sequence, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
import itertools
from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, utils
from src.dataset.video_transform import RGBtoYUVTransform

def _rgb_from_proto(proto) -> torch.Tensor:
    img_bgr = cv2.imdecode(np.frombuffer(proto.image, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0

def _project_top_lidar(
    frame,
    camera_name,
    lidar_name
) -> Tuple[torch.Tensor, torch.Tensor]:

    img_proto = next(i for i in frame.images if i.name == camera_name)
    rgb = _rgb_from_proto(img_proto)

    _, H, W = rgb.shape

    laser = next(l for l in frame.lasers if l.name == lidar_name)
    ri, cam_proj, _ = utils.parse_range_image_and_camera_projection(laser)
    pcl, _ = utils.project_to_pointcloud(
        frame,
        ri,
        cam_proj,
        _,
        utils.get(frame.context.laser_calibrations, lidar_name),
    )

    # Keep only pixels with valid LiDAR returns
    valid_mask = ri[..., 0] > 0
    pcl = pcl.reshape(-1, 3)  # (N, 3)
    cam_proj = cam_proj.reshape(-1, 6)[valid_mask.reshape(-1)]  # (N, 6)

    # Extract point indices that fall inside the front camera FOV
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
    front_pts = pcl[front_mask]

    cam_cal = next(
        c for c in frame.context.camera_calibrations if c.name == camera_name
    )
    T_c2v = np.asarray(cam_cal.extrinsic.transform, dtype=np.float32).reshape(4, 4)  # C→V
    T_v2c = np.linalg.inv(T_c2v)  # Vehicle → Camera

    # Channel 0 = depth (X_cam)
    pts_cam = (T_v2c @ np.c_[front_pts, np.ones(len(front_pts))].T).T
    depth_cam = pts_cam[:, 0]

    intensity = ri[..., 1][valid_mask][front_mask].astype(np.float32)
    elong = ri[..., 2][valid_mask][front_mask].astype(np.float32)

    proj = np.full((3, H, W), fill_value=0, dtype=np.float32)
    proj[0, v_px, u_px] = depth_cam / 75.0
    proj[1, v_px, u_px] = np.clip(intensity, 0, 1.5) / 1.5
    proj[2, v_px, u_px] = elong / 1.5

    return torch.as_tensor(proj, dtype=torch.float32), rgb

class WaymoDataset(Dataset):

    OFFSETS_ATTRS: Tuple[str, ...] = ("_table", "_frame_table")

    def __init__(
        self,
        tfrecord_paths: Sequence[str] | str,
        seq_len: int = 1,
        slide: int = 1,
        crop_size: int = None,
        transform=None,
        yuv_format=None,
    ) -> None:
        super().__init__()

        if isinstance(tfrecord_paths, str):
            tfrecord_paths = sorted(glob.glob(tfrecord_paths))
        self.tfrecord_paths = list(tfrecord_paths)
        print(self.tfrecord_paths)
        if not self.tfrecord_paths:
            raise ValueError("No TFRecord files found.")

        self.seq_len = seq_len
        self.slide = slide
        self.crop_size = crop_size
        self.transform = transform
        self.yuv_transform = RGBtoYUVTransform(yuv_format) if yuv_format else None

        self.camera_name = dataset_pb2.CameraName.FRONT
        self.lidar_name =  dataset_pb2.LaserName.TOP
        # Build (path, start) index.
        self._index: List[Tuple[str, int]] = []
        for p in self.tfrecord_paths:
            total = self._count_frames(p)
            if total < seq_len:
                continue
            self._index.extend((p, s) for s in range(0, total - seq_len + 1, slide))

        self._reader_cache: Dict[str, WaymoDataFileReader] = {}

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, start = self._index[idx]
        reader = self._get_reader(path)
        offsets_attr = next((a for a in self.OFFSETS_ATTRS if hasattr(reader, a)), None)
        if offsets_attr is not None:
            offsets = getattr(reader, offsets_attr)
            proj_seq, rgb_seq = [], []
            for i in range(start, start + self.seq_len):
                reader.file_handle.seek(offsets[i])
                frame = reader._read_frame()
                proj, rgb = _project_top_lidar(frame, self.camera_name)
                proj_seq.append(proj)
                rgb_seq.append(rgb)
        else:
            reader.seek(0)
            frames = itertools.islice(reader, start, start + self.seq_len)
            proj_seq, rgb_seq = zip(*(_project_top_lidar(f, self.camera_name, self.lidar_name) for f in frames))

        if self.transform is not None:
            proj_seq = [self.transform(p) for p in proj_seq]
            rgb_seq  = [self.transform(r) for r in rgb_seq]

        # random crop
        if self.crop_size is not None:
            crop_h, crop_w = int(self.crop_size), int(self.crop_size)
            _, H, W = proj_seq[0].shape
            if crop_h > H or crop_w > W:
                raise ValueError("crop_size larger than input size")
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            proj_seq = [p[:, top:top + crop_h, left:left + crop_w] for p in proj_seq]
            rgb_seq  = [r[:, top:top + crop_h, left:left + crop_w] for r in rgb_seq]

        if self.yuv_transform:
            rgb_seq  = [self.yuv_transform(r) for r in rgb_seq]

        return  torch.stack(list(proj_seq), 0), torch.stack(list(rgb_seq), 0)

    def _get_reader(self, path: str) -> WaymoDataFileReader:
        info = get_worker_info()
        cache: Dict[str, WaymoDataFileReader] = (
            self._reader_cache if info is None else info.dataset._reader_cache  # type: ignore[attr-defined]
        )
        if path in cache:
            return cache[path]

        reader = WaymoDataFileReader(path)

        # Ensure that at least one offsets attribute exists if possible.
        if not any(hasattr(reader, a) for a in self.OFFSETS_ATTRS):
            for _ in reader:  # full pass to build internal tables if implemented
                pass
            reader.seek(0)
        cache[path] = reader
        return reader

    @staticmethod
    def _count_frames(path: str) -> int:
        return sum(1 for _ in WaymoDataFileReader(path))

# build_mask_cache.py
# Generate & store YOLO segmentation masks for Waymo TFRecords.
# Supported storage formats: "png" (binary 0/255) or "npz" (binary uint8 0/1).
#
# Usage (example):
#   from build_mask_cache import build_cache
#   build_cache(
#       tf_glob="/data/waymo/*.tfrecord",
#       cache_dir="seg_cache",
#       yolo_weights="/path/to/yolov8s-seg.pt",
#       storage_format="npz",          # or "png"
#       classes_keep=None,             # e.g., [2, 5] to keep only person+car
#       thr=0.5,
#       min_area=64,                   # discard tiny fragments
#       morph="open", morph_ksize=3,   # None | "open" | "close" | "erode" | "dilate"
#       device="cuda"                  # or "cpu"
#   )

import os, glob, json, time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from ultralytics import YOLO
from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2

# ------------------------------- I/O helpers -------------------------------

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def out_path(cache_dir: str | Path, tf_path: str | Path, frame_idx: int, storage_format: str) -> Path:
    base = Path(tf_path).stem
    d = Path(cache_dir) / base
    ensure_dir(d)
    ext = ".png" if storage_format == "png" else ".npz"
    return d / f"{frame_idx:06d}{ext}"

def write_mask_png(path: Path, mask_01: np.ndarray) -> None:
    # mask_01: (H,W) uint8 {0,1}
    tmp = str(path) + ".tmp.png"
    cv2.imwrite(tmp, (mask_01 * 255).astype(np.uint8))
    os.replace(tmp, str(path))

def write_mask_npz(path: Path, mask_01: np.ndarray, meta: dict | None = None) -> None:
    # mask_01: (H,W) uint8 {0,1}
    # we store 'mask' and (optionally) minimal metadata
    arrays = {"mask": mask_01.astype(np.uint8)}
    if meta:
        # flatten meta to types np.savez can serialize
        flat_meta = {f"meta_{k}": np.array(v) for k, v in meta.items()}
        arrays.update(flat_meta)
    tmp = str(path) + ".tmp"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp + ".npz", str(path))

def read_rgb_from_frame(frame, camera_name) -> np.ndarray:
    # Returns HWC RGB uint8
    img_proto = next(i for i in frame.images if i.name == camera_name)
    arr = np.frombuffer(img_proto.image, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

# ---------------------------- mask post-processing ----------------------------

def _union_instance_masks(
    res,
    H: int,
    W: int,
    classes_keep: Optional[Sequence[int]] = None,
    thr: float = 0.5,
    min_area: int = 0,
    morph: Optional[str] = None,
    morph_ksize: int = 3,
) -> np.ndarray:
    """
    Build a single binary mask (H,W) uint8 in {0,1} from an Ultralytics result:
      - optional class filtering
      - threshold on logits
      - optional min area filtering
      - optional morphology (open/close/erode/dilate)
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    # No masks predicted
    if getattr(res, "masks", None) is None or res.masks is None or res.masks.data is None:
        return mask

    data = res.masks.data  # (N, H', W') float/bool
    if data is None or data.numel() == 0:
        return mask

    data_np = data.detach().cpu().numpy()  # (N, H', W')
    # If shapes don't match original size, resize to (H,W)
    h2, w2 = data_np.shape[-2:]
    if (h2, w2) != (H, W):
        # nearest preserves binary edges
        data_np = np.stack([cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                            for m in data_np], axis=0)

    # Class filter (optional)
    if classes_keep is not None and getattr(res, "boxes", None) is not None and res.boxes is not None and res.boxes.cls is not None:
        cls = res.boxes.cls.detach().cpu().numpy().astype(int)  # (N,)
        keep = np.isin(cls, np.array(classes_keep, dtype=int))
        data_np = data_np[keep]
        if data_np.size == 0:
            return mask

    # Threshold → union
    union = (data_np > float(thr)).any(axis=0).astype(np.uint8)  # (H,W) {0,1}

    # Min area filter (remove small blobs)
    if min_area > 0 and union.any():
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=4)
        keep = np.zeros_like(union)
        for i in range(1, nb):  # 0 is background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                keep[labels == i] = 1
        union = keep

    # Morphology (optional)
    if morph:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
        if morph == "open":
            union = cv2.morphologyEx(union, cv2.MORPH_OPEN, k)
        elif morph == "close":
            union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, k)
        elif morph == "erode":
            union = cv2.erode(union, k, iterations=1)
        elif morph == "dilate":
            union = cv2.dilate(union, k, iterations=1)
        else:
            raise ValueError(f"Unknown morph op: {morph}")

    return union.astype(np.uint8)

# ------------------------------- main builder -------------------------------

def build_cache(
    tf_glob: str,
    cache_dir: str = "seg_cache",
    yolo_weights: str = "yolov8x-seg.pt",
    storage_format: str = "npz",   # "npz" or "png"
    camera_name = dataset_pb2.CameraName.FRONT,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "cuda",          # "cuda" or "cpu"
    classes_keep: Optional[Sequence[int]] = None,
    thr: float = 0.5,
    min_area: int = 0,
    morph: Optional[str] = None,
    morph_ksize: int = 3,
    overwrite: bool = False,
    progress_every: int = 50,
) -> None:
    """
    Iterate all frames in matched TFRecords, generate UNION binary masks, and store them.

    Output file per frame:
      seg_cache/<tfrecord_basename>/<frame_idx>.npz  (if storage_format="npz", contains 'mask')
      seg_cache/<tfrecord_basename>/<frame_idx>.png  (if storage_format="png",  0/255 image)

    You can re-run safely; existing files are skipped unless overwrite=True.
    """
    assert storage_format in ("npz", "png"), "storage_format must be 'npz' or 'png'"
    ensure_dir(cache_dir)

    files = sorted(glob.glob(tf_glob))
    if not files:
        raise FileNotFoundError(f"No TFRecords matched: {tf_glob}")

    # YOLO model in the main process
    model = YOLO(yolo_weights)
    model.to(device)
    model.fuse()
    model.eval()

    total_frames = 0
    written = 0
    t0 = time.time()

    for tf_path in files:
        base = Path(tf_path).stem
        print(f"[cache] processing {base} ...")
        rdr = WaymoDataFileReader(tf_path)
        for idx, frame in enumerate(rdr):
            total_frames += 1
            out = out_path(cache_dir, tf_path, idx, storage_format)
            if out.exists() and not overwrite:
                continue

            rgb = read_rgb_from_frame(frame, camera_name)  # HWC uint8
            H, W = rgb.shape[:2]

            # One-frame predict (keeps logic simple & robust to image sizes)
            res = model.predict(
                rgb,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False
            )[0]

            mask01 = _union_instance_masks(
                res, H, W,
                classes_keep=classes_keep,
                thr=thr,
                min_area=min_area,
                morph=morph,
                morph_ksize=morph_ksize
            )  # (H,W) uint8 0/1

            if storage_format == "png":
                write_mask_png(out, mask01)
            else:
                meta = {
                    "H": int(H), "W": int(W),
                    "thr": float(thr),
                    "min_area": int(min_area),
                    "morph": morph or "",
                    "weights": str(Path(yolo_weights).name),
                }
                write_mask_npz(out, mask01, meta=meta)

            written += 1
            if progress_every and written % progress_every == 0:
                dt = time.time() - t0
                print(f"  wrote {written} masks in {dt:.1f}s")

    print(f"[cache] done: wrote {written}/{total_frames} ({storage_format}) in {time.time()-t0:.1f}s")

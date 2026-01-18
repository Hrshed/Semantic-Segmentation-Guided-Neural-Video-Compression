# waymo_dataset_lightning.py

import glob, math, os, random
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
from src.dataset.seg_waymo_dataset import WaymoDataset

# --- NEW: seed each worker deterministically (helps reproduce bad samples) ---
def _seed_worker(worker_id: int):
    base = torch.initial_seed() % 2**32
    np.random.seed(base)
    random.seed(base)

# --- NEW: strict collate to fail early on shape/dtype/layout surprises ---
from torch.utils.data._utils.collate import default_collate

def _strict_collate(batch):
    """Accepts dicts or tuples/lists from the dataset; validates shapes and contiguity."""
    first = batch[0]

    # Case A: dict samples
    if isinstance(first, dict):
        keys = first.keys()
        for b in batch:
            if not isinstance(b, dict) or b.keys() != keys:
                raise RuntimeError(f"Mismatched keys in batch: got {getattr(b,'keys',lambda: 'N/A')()} vs {keys}")

        out = {}
        for k in keys:
            elems = [b[k] for b in batch]
            if torch.is_tensor(elems[0]):
                # shape must match across the batch
                shapes = [tuple(e.shape) for e in elems]
                if any(s != shapes[0] for s in shapes):
                    raise RuntimeError(f"Variable shapes for key '{k}': {shapes}")
                elems = [e.contiguous() for e in elems]
            out[k] = default_collate(elems)
        return out

    # Case B: tuple/list samples  (e.g., (proj_seq, ycbcrm_seq))
    if isinstance(first, (tuple, list)):
        n = len(first)
        for b in batch:
            if not isinstance(b, (tuple, list)) or len(b) != n:
                raise RuntimeError(f"Mixed or ragged tuple samples in batch: lens {[len(x) if isinstance(x,(tuple,list)) else type(x) for x in batch]}")
        collated = []
        for i in range(n):
            elems = [b[i] for b in batch]
            if torch.is_tensor(elems[0]):
                shapes = [tuple(e.shape) for e in elems]
                if any(s != shapes[0] for s in shapes):
                    raise RuntimeError(f"Variable shapes at tuple index {i}: {shapes}")
                elems = [e.contiguous() for e in elems]
            collated.append(default_collate(elems))
        return tuple(collated)  # preserve your original structure

    # Fallback: plain tensors/ndarrays/etc.
    return default_collate(batch)



class WaymoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tf_glob: str,
        seg_cache_dir: str = "seg_cache",
        seq_len: int = 8,
        slide: int = 1,
        crop_size: int | None = 256,
        train_val_test: tuple[float, float, float] = (0.8, 0.2, 0.0),
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        strict_masks: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.tf_glob = tf_glob
        self.seg_cache_dir = seg_cache_dir
        self.seq_len = int(seq_len)
        self.slide = int(slide)
        self.crop_size = crop_size
        self.train_val_test = train_val_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.strict_masks = strict_masks
        self.seed = seed

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        # --- NEW: safer defaults in the parent process ---
        os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
        try:
            # avoids "too many open files" & shared-memory weirdness on long runs
            torch.multiprocessing.set_sharing_strategy("file_system")
        except RuntimeError:
            pass
        # keep OpenCV from spinning per-worker thread pools
        cv2.setNumThreads(0)

    def setup(self, stage: str | None = None):
        files = sorted(glob.glob(self.tf_glob))
        if not files:
            raise ValueError(f"No TFRecords matched: {self.tf_glob}")

        full_ds = WaymoDataset(
            tfrecord_paths=files,
            seg_cache_dir=self.seg_cache_dir,
            seq_len=self.seq_len,
            slide=self.slide,
            crop_size=self.crop_size,
            strict_masks=self.strict_masks,
        )

        n = len(full_ds)
        tvt = self.train_val_test
        if not math.isclose(sum(tvt), 1.0, rel_tol=1e-6):
            raise ValueError(f"train_val_test must sum to 1.0, got {tvt}")
        n_train = int(n * tvt[0])
        n_val = int(n * tvt[1])
        n_test = n - n_train - n_val

        g = torch.Generator().manual_seed(self.seed)
        all_indices = torch.randperm(n, generator=g).tolist()
        train_idx = all_indices[:n_train]
        val_idx = all_indices[n_train:n_train + n_val]
        test_idx = all_indices[n_train + n_val:] if n_test > 0 else []

        if stage in (None, "fit"):
            self.train_dataset = Subset(full_ds, train_idx) if n_train > 0 else None
            self.val_dataset = Subset(full_ds, val_idx) if n_val > 0 else None

        if stage in (None, "test"):
            self.test_dataset = Subset(full_ds, test_idx) if n_test > 0 else None

    # Helper to build consistent loader kwargs
    def _loader_kwargs(self, shuffle: bool):
        kw = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
            drop_last=False,                 # set True if you need fixed batch shapes
            worker_init_fn=_seed_worker,     # NEW
            collate_fn=_strict_collate,      # NEW
        )
        if self.num_workers > 0:
            # NEW: calm down concurrency between CPU->GPU transfers
            kw["prefetch_factor"] = 1
        return kw

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))

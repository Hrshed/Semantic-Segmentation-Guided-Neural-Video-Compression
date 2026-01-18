# waymo_dataset_lightning.py
# Lightning DataModule that wraps the read-only WaymoDataset (YCbCr + mask).
# Returns (proj_seq, ycbcrm_seq) where ycbcrm = [Y, Cb, Cr, Mask] in [0,1].

from __future__ import annotations

import glob
import math
import os
import random
from typing import List, Optional, Sequence, Tuple

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset.seg_waymo_dataset import WaymoDataset

class WaymoDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Waymo + cached masks.

    Args:
        tf_glob:          Glob for TFRecord files, e.g. "/data/waymo/*.tfrecord"
        seg_cache_dir:    Folder where masks are cached: <seg_cache>/<basename>/<frame_idx>.(npz|png)
        seq_len:          Sequence length per sample
        slide:            Stride for sequence windowing
        crop_size:        Square crop applied identically to proj/img/mask (None = no crop)
        train_val_test:   Fractions that sum to 1.0
        batch_size:       Batch size for all splits (set per-split below if you prefer)
        num_workers:      DataLoader workers
        pin_memory:       Pin memory for speed on GPU
        strict_masks:     If True, raise if any mask file is missing; else fill zeros
        seed:             Seed for deterministic split/shuffle
    """

    def __init__(
        self,
        tf_glob: str,
        seg_cache_dir: str = "seg_cache",
        seq_len: int = 8,
        slide: int = 1,
        crop_size: Optional[int] = 256,
        train_val_test: Tuple[float, float, float] = (0.8, 0.2, 0.0),
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

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        files = sorted(glob.glob(self.tf_glob))
        if not files:
            raise ValueError(f"No TFRecords matched: {self.tf_glob}")

        # Build a single full dataset first; we'll split by indices
        full_ds = WaymoDataset(
            tfrecord_paths=files,
            seg_cache_dir=self.seg_cache_dir,
            seq_len=self.seq_len,
            slide=self.slide,
            crop_size=self.crop_size,
            strict_masks=self.strict_masks,
        )

        # Deterministic split
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

    # Dataloaders
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
            drop_last=False,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0),
            drop_last=False,
        )

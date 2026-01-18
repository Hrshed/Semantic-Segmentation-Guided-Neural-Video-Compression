
import glob
import os
import random
from typing import Optional, Sequence
from dataset.waymo_dataset import WaymoDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

class WaymoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 4,
        slide: int = 1,
        crop_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ):
        super().__init__()
        self.datadir = data_dir
        self.seq_len = seq_len
        self.slide = slide
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.transform = None # Placeholder for any torchvision transforms
        self.yuv_format = 444 # e.g., 'YUV420'

        # Datasets will be assigned in setup()
        self.train_dataset: Optional[WaymoDataset] = None
        self.val_dataset: Optional[WaymoDataset] = None
        self.test_dataset: Optional[WaymoDataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        Splits the dataset into train, validation, and test sets.
        This method is called by PyTorch Lightning.
        """

        all_files = glob.glob(os.path.join(self.datadir, "*.tfrecord"))
        print(f"Found {len(all_files)} tfrecord files in {self.datadir}")

        random.shuffle(all_files)

        num_files = len(all_files)
        num_train = int(self.train_val_test_split[0] * num_files)
        num_val = int(self.train_val_test_split[1] * num_files)
        num_test = num_files - num_train - num_val

        if num_train + num_val + num_test != num_files:
            raise ValueError("Train/val/test split percentages do not sum to 1.")

        train_files = all_files[:num_train]
        val_files = all_files[num_train:num_train + num_val]
        test_files = all_files[num_train + num_val:]

        if stage == "fit" or stage is None:
            self.train_dataset = WaymoDataset(
                tfrecord_paths=train_files,
                seq_len=self.seq_len,
                slide=self.slide,
                crop_size=self.crop_size,
                transform=self.transform,
                yuv_format=self.yuv_format,
            )
            self.val_dataset = WaymoDataset(
                tfrecord_paths=val_files,
                seq_len=self.seq_len,
                slide=self.slide,
                crop_size=self.crop_size,
                transform=self.transform,
                yuv_format=self.yuv_format,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WaymoDataset(
                tfrecord_paths=test_files,
                seq_len=self.seq_len,
                slide=self.slide,
                crop_size=self.crop_size,
                transform=self.transform,
                yuv_format=self.yuv_format,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not configured. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not configured. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not configured. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
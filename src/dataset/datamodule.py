import os
import random
from typing import Optional, Sequence, Union, List
import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.dataset.waymo_dataset import WaymoDataset
from src.dataset.dataset_vimeo import (
    Vimeo90kDataset,
    Vimeo90kMP4Dataset,
    Vimeo90kSeptupletDataset,
    Vimeo90kImageDataset
)

class UnifiedVideoDataModule(pl.LightningDataModule):
    """
    Unified DataModule supporting multiple video datasets:
    - Waymo: 'waymo'
    - Vimeo Septuplet (main): 'vimeo_septuplet'
    - Vimeo Image sequences: 'vimeo_image'
    - Vimeo MP4: 'vimeo_mp4'
    - Vimeo Individual Images: 'vimeo_single_image'
    """

    def __init__(
        self,
        dataset_type: str,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 4,

        # Video/sequence parameters
        n_frames: int = 7,
        seq_len: Optional[int] = None,  # For Waymo compatibility
        slide: int = 1,

        # Image processing
        crop: Optional[Union[int, List[int]]] = None,
        crop_size: Optional[int] = None,  # For Waymo compatibility
        yuv_format: str = "444",

        # Transforms
        transform: Optional[callable] = None,
        sequence_transform: Optional[callable] = None,

        # Waymo specific
        train_val_test_split: tuple = (0.8, 0.1, 0.1),

        # Vimeo MP4 specific
        generate_split: bool = False,
        train_split: float = 0.8,
        use_cache: bool = True,

        # Additional paths
        video_dir: str = "",
        train_data_path: Optional[str] = None,
        val_data_path: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Normalize frame parameters
        self.n_frames = seq_len or n_frames  # Use seq_len for Waymo, n_frames for Vimeo
        self.slide = slide

        # Normalize crop parameters
        if crop_size is not None:
            self.crop = crop_size
        elif isinstance(crop, int):
            self.crop = crop
        elif isinstance(crop, (list, tuple)) and len(crop) == 2:
            self.crop = crop
        else:
            self.crop = crop

        self.yuv_format = yuv_format
        self.transform = transform
        self.sequence_transform = sequence_transform

        # Waymo specific
        self.train_val_test_split = train_val_test_split

        # Vimeo MP4 specific
        self.generate_split = generate_split
        self.train_split = train_split
        self.use_cache = use_cache
        self.video_dir = video_dir or data_dir

        # Override paths
        self.train_data_path = train_data_path or data_dir
        self.val_data_path = val_data_path or data_dir

        # Validate dataset type
        valid_types = [
            "waymo", "vimeo_septuplet", "vimeo_image",
            "vimeo_mp4", "vimeo_single_image"
        ]
        if dataset_type not in valid_types:
            raise ValueError(f"dataset_type must be one of {valid_types}")

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets based on dataset_type"""

        if self.dataset_type == "waymo":
            self._setup_waymo_datasets(stage)
        elif self.dataset_type == "vimeo_septuplet":
            self._setup_vimeo_septuplet_datasets(stage)
        elif self.dataset_type == "vimeo_image":
            self._setup_vimeo_image_datasets(stage)
        elif self.dataset_type == "vimeo_mp4":
            self._setup_vimeo_mp4_datasets(stage)
        elif self.dataset_type == "vimeo_single_image":
            self._setup_vimeo_single_image_datasets(stage)

    def _setup_waymo_datasets(self, stage: Optional[str] = None):
        """Setup Waymo datasets with tfrecord files"""
        all_files = glob.glob(os.path.join(self.data_dir, "*.tfrecord"))
        all_files = [os.path.abspath(f) for f in all_files]
        print(f"Found {len(all_files)} tfrecord files in {self.data_dir}")

        if not all_files:
            raise ValueError("No tfrecord files found.")

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
                seq_len=self.n_frames,
                slide=self.slide,
                crop_size=self.crop,
                transform=self.transform,
                yuv_format=self.yuv_format,
            )

            self.val_dataset = WaymoDataset(
                tfrecord_paths=val_files,
                seq_len=self.n_frames,
                slide=self.slide,
                crop_size=self.crop,
                transform=self.transform,
                yuv_format=self.yuv_format,
            )

    def _setup_vimeo_septuplet_datasets(self, stage: Optional[str] = None):
        """Setup Vimeo Septuplet datasets (main Vimeo dataset)"""
        if stage == "fit" or stage is None:
            self.train_dataset = Vimeo90kSeptupletDataset(
                root_dir=self.train_data_path,
                mode="train",
                n_frames=self.n_frames,
                transform=self.transform,
                crop=self.crop,
                yuv_format=self.yuv_format,
                sequence_transform=self.sequence_transform,
            )

            self.val_dataset = Vimeo90kSeptupletDataset(
                root_dir=self.val_data_path,
                mode="test",
                n_frames=self.n_frames,
                transform=self.transform,
                crop=self.crop,
                yuv_format=self.yuv_format,
                sequence_transform=self.sequence_transform,
            )

    def _setup_vimeo_image_datasets(self, stage: Optional[str] = None):
        """Setup Vimeo image sequence datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = Vimeo90kDataset(
                root_dir=self.train_data_path,
                mode="train",
                n_frames=self.n_frames,
                transform=self.transform,
                crop=self.crop,
                yuv_format=self.yuv_format,
                sequence_transform=self.sequence_transform,
            )

            self.val_dataset = Vimeo90kDataset(
                root_dir=self.val_data_path,
                mode="test",
                n_frames=self.n_frames,
                transform=self.transform,
                crop=self.crop,
                yuv_format=self.yuv_format,
                sequence_transform=self.sequence_transform,
            )

    def _setup_vimeo_mp4_datasets(self, stage: Optional[str] = None):
        """Setup Vimeo MP4 datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = Vimeo90kMP4Dataset(
                video_dir=self.video_dir,
                n_frames=self.n_frames,
                transform=self.transform,
                crop=self.crop,
                yuv_format=self.yuv_format,
                sequence_transform=self.sequence_transform,
                mode="train",
                generate_split=self.generate_split,
                train_split=self.train_split,
                use_cache=self.use_cache,
            )

            self.val_dataset = Vimeo90kMP4Dataset(
                video_dir=self.video_dir,
                n_frames=self.n_frames,
                transform=self.transform,
                crop=self.crop,
                yuv_format=self.yuv_format,
                sequence_transform=self.sequence_transform,
                mode="test",
                generate_split=self.generate_split,
                train_split=self.train_split,
                use_cache=self.use_cache,
            )

    def _setup_vimeo_single_image_datasets(self, stage: Optional[str] = None):
        """Setup Vimeo individual image datasets"""
        if stage == "fit" or stage is None:
            # Convert crop to proper format for image dataset
            crop_size = self.crop if isinstance(self.crop, (list, tuple)) else (self.crop, self.crop)

            self.train_dataset = Vimeo90kImageDataset(
                data_dir=self.train_data_path,
                mode="train",
                crop_size=crop_size,
                transform=self.transform,
            )

            self.val_dataset = Vimeo90kImageDataset(
                data_dir=self.val_data_path,
                mode="test",
                crop_size=crop_size,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not configured. Call setup() first.")

        # Determine shuffle and drop_last based on dataset type
        shuffle = True
        drop_last = self.dataset_type == "waymo"  # Waymo typically needs drop_last=True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not configured. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not configured. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    @property
    def num_classes(self) -> int:
        """Return number of classes if applicable"""
        return 0  # Not applicable for video compression tasks

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dataset_type={self.dataset_type}, "
            f"data_dir={self.data_dir}, "
            f"batch_size={self.batch_size}, "
            f"n_frames={self.n_frames})"
        )
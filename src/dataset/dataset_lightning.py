from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .dataset_vimeo import Vimeo90kDataset, Vimeo90kMP4Dataset


# --- VideoDataModule ---
class VideoDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule to load video data.

    Depending on the 'dataset_type' parameter, it instantiates either:
      - an image sequence dataset (using your provided Vimeo90kDataset from :contentReference[oaicite:3]{index=3})
      - or an MP4 dataset (using the MP4VideoDataset defined above).

    Args:
        dataset_type (str): 'image' for image sequences or 'mp4' for MP4 videos.
        data_dir (str): Root directory of the dataset.
            For 'image', this should be the root used by Vimeo90kDataset (which contains the "sequences" folder and the list file).
            For 'mp4', this should be a folder containing MP4 files.
        batch_size (int): Batch size.
        n_frames (int): Number of frames per sample.
        transform (callable, optional): Transform to apply to each frame.
        num_workers (int): Number of DataLoader workers.
    """

    def __init__(
        self,
        dataset_type,
        data_dir,
        batch_size=4,
        n_frames=7,
        transform=None,
        crop=None,
        yuv_format="444",
        sequence_transform=None,
        mode="train",
        num_workers=4,
        video_dir="",
    ):
        super().__init__()
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.transform = transform
        self.num_workers = num_workers
        self.crop = crop
        self.yuv_format = yuv_format
        self.sequence_transform = sequence_transform
        self.mode = mode
        self.video_dir = video_dir

    def setup(self, stage=None):
        if self.dataset_type == "image":
            self.train_dataset = Vimeo90kDataset(
                root_dir=self.data_dir,
                mode="train",
                n_frames=self.n_frames,
                crop=self.crop,
                yuv_format=self.yuv_format,
                sequence_transform=self.sequence_transform,
            )
            self.val_dataset = Vimeo90kDataset(
                root_dir=self.data_dir,
                mode="test",
                n_frames=self.n_frames,
                crop=self.crop,
                yuv_format=self.yuv_format,
                sequence_transform=self.sequence_transform,
            )
        elif self.dataset_type == "mp4":
            self.train_dataset = Vimeo90kMP4Dataset(
                video_dir=self.data_dir,
                n_frames=self.n_frames,
                crop=self.crop,
                yuv_format=self.yuv_format,
                sequence_transform=self.sequence_transform,
                mode="train",
            )
            self.val_dataset = Vimeo90kMP4Dataset(
                video_dir=self.data_dir,
                n_frames=self.n_frames,
                mode="test",
                crop=self.crop,
                yuv_format=self.yuv_format,
                sequence_transform=self.sequence_transform,
            )
        else:
            raise ValueError("dataset_type must be 'image' or 'mp4'.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

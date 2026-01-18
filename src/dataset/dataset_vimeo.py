import time
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
from .video_transform import RandomCropTransform, RGBtoYUVTransform
import cv2
import threading
import random


class Vimeo90kImageDataset(Dataset):
    """
    A dataset that treats each frame from the Vimeo90k sequences as an
    individual image sample. It randomly samples images across all sequences
    defined in the corresponding train/test list file.

    Instead of resizing, it performs a random crop to the target size.
    If an image is smaller than the target crop size, it's first resized
    to the crop size before cropping (effectively taking the whole resized image).

    Args:
        data_dir (str): The root directory containing the 'sequences' folder
                        and the 'sep_trainlist.txt'/'sep_testlist.txt' files.
        mode (str): 'train' or 'test' to load the respective list file.
        crop_size (tuple): The target (height, width) for the random crop.
        transform (callable, optional): Optional transform to be applied
                                        on a PIL image sample *after* cropping
                                        but *before* converting to a Tensor.
                                        Useful for augmentations like
                                        RandomHorizontalFlip, ColorJitter etc.
                                        Note: ToTensor is applied *after* this transform.
    """

    def __init__(self, data_dir, mode="train", crop_size=(256, 256), transform=None):
        self.data_dir = data_dir
        self.mode = mode
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        self.to_tensor_transform = transforms.ToTensor()
        self.random_crop_transform = transforms.RandomCrop(self.crop_size)
        self.custom_transform = transform

        list_filename = "sep_trainlist.txt" if mode == "train" else "sep_testlist.txt"
        list_path = os.path.join(data_dir, list_filename)

        self.image_paths = []
        if os.path.exists(list_path):
            with open(list_path, "r") as f:
                sequence_folders = f.read().splitlines()

            for seq_folder in sequence_folders:
                sequence_path = os.path.join(data_dir, "sequences", seq_folder)
                if os.path.isdir(sequence_path):
                    for i in range(1, 8):
                        image_name = f"im{i}.png"
                        image_path = os.path.join(sequence_path, image_name)
                        if os.path.exists(image_path):
                            self.image_paths.append(image_path)

        else:
            raise RuntimeError(f"List file {list_path} does not exist.")

        if not self.image_paths:
            raise RuntimeError(f"No image files found for mode '{mode}' in {data_dir}. Check list file and sequences folder.")

        print(f"Found {len(self.image_paths)} individual images in {data_dir} for mode '{mode}'. Target crop size: {self.crop_size}")

    def __len__(self):
        """Returns the total number of individual images found."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads, potentially resizes, randomly crops, transforms, and returns a single image.

        Args:
            idx (int): Index of the image file path in the flattened list.

        Returns:
            torch.Tensor: The transformed image tensor (C, H, W).
        """
        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
            crop_height, crop_width = self.crop_size

            if img_height < crop_height or img_width < crop_width:
                resize_transform = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.BILINEAR)
                image = resize_transform(image)
            image = self.random_crop_transform(image)
            image_tensor = self.to_tensor_transform(image)
            if self.custom_transform:
                image = self.custom_transform(image)
            if image_tensor.shape[1] != crop_height or image_tensor.shape[2] != crop_width:
                print(
                    f"Warning: Tensor shape mismatch for {image_path}. Expected (3, {crop_height}, {crop_width}), got {image_tensor.shape}. Resizing tensor."
                )
                image_tensor = transforms.functional.resize(image_tensor, list(self.crop_size), antialias=True)
            return image_tensor
        except Exception as e:
            print(f"Error loading or processing image {image_path}: {e}")
            return torch.zeros((3, self.crop_size[0], self.crop_size[1]))


class Vimeo90kSeptupletDataset(Dataset):
    """
    Dataset for loading Vimeo-90k septuplet data, where each sequence is a
    single MP4 file with the directory structure:
    .../sequences/<video_id>/<sequence_id>/<sequence_id>.mp4
    """

    def __init__(
        self,
        root_dir,
        mode="train",
        n_frames=7,
        transform=None,
        crop=None,
        yuv_format="420",
        sequence_transform=None,
    ):
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.base_transform = transforms.ToTensor()
        self.transform = transforms.Compose([transform, self.base_transform]) if transform else self.base_transform

        if isinstance(crop, int):
            crop = (crop, crop)
        self.crop = crop

        self.yuv_transform = RGBtoYUVTransform(yuv_format) if yuv_format else None
        self.sequence_transform = sequence_transform

        # Load sequence list from sep_trainlist.txt or sep_testlist.txt
        list_filename = f"sep_{mode}list.txt"
        list_path = os.path.join(self.root_dir, list_filename)
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")

        with open(list_path, "r") as f:
            sequence_paths = f.read().splitlines()

        self.video_files = []

        print(f"Locating video files for mode '{mode}'...")
        for seq_path in sequence_paths:
            # e.g., seq_path = '00001/0010'
            video_folder_name = os.path.basename(seq_path)  # '0010'
            video_filename = f"{video_folder_name}.mp4"  # '0010.mp4'
            full_video_path = os.path.join(self.root_dir, "sequences", seq_path, video_filename)

            if os.path.exists(full_video_path):
                self.video_files.append(full_video_path)

        if not self.video_files:
            raise RuntimeError(f"No video files found for mode '{mode}'. Check paths and directory structure.")

        print(f"Found {len(self.video_files)} video files for mode '{mode}'.")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < self.n_frames:
            cap.release()
            raise RuntimeError(f"Video '{video_path}' has only {total_frames} frames, but {self.n_frames} are required.")

        # Randomly select a starting frame for the sequence
        start_frame = random.randint(0, total_frames - self.n_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        rgb_images = []
        yuv_images = []
        crop_trans = None

        for i in range(self.n_frames):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError(f"Failed to read frame {start_frame + i} from {video_path}.")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_width, img_height = pil_img.size

            rgb_tensor = self.transform(pil_img)

            if self.crop and crop_trans is None:
                crop_trans = RandomCropTransform(self.crop[0], self.crop[1], img_width, img_height)

            if crop_trans:
                rgb_tensor = crop_trans(rgb_tensor)

            rgb_images.append(rgb_tensor)

            if self.yuv_transform:
                yuv_tensor = self.yuv_transform(rgb_tensor.unsqueeze(0))
                if isinstance(yuv_tensor, tuple):  # YUV420
                    y_plane, uv_plane = yuv_tensor[0].squeeze(0), yuv_tensor[1].squeeze(0)
                    yuv_images.append((y_plane, uv_plane))
                else:  # YUV444
                    yuv_images.append(yuv_tensor.squeeze(0))
        cap.release()

        rgb_sequence = torch.stack(rgb_images, dim=0).contiguous()
        output = {"rgb": rgb_sequence}

        if self.yuv_transform:
            if isinstance(yuv_images[0], tuple): # YUV420
                y_frames = torch.stack([img[0] for img in yuv_images], dim=0).contiguous()
                uv_frames = torch.stack([img[1] for img in yuv_images], dim=0).contiguous()
                output["yuv"] = (y_frames, uv_frames)
            else: # YUV444
                yuv_sequence = torch.stack(yuv_images, dim=0).contiguous()
                output["yuv"] = yuv_sequence

        if self.sequence_transform:
            output = self.sequence_transform(output)

        return output


class Vimeo90kDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode="test",
        n_frames=7,
        transform=None,
        crop=None,
        yuv_format="444",
        sequence_transform=None,
    ):
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.transform = transforms.ToTensor()
        if transform is not None:
            self.transform = transforms.Compose([transform, self.transform])
        self.crop = crop
        self.yuv_transform = RGBtoYUVTransform(yuv_format) if yuv_format else None

        list_filename = "sep_trainlist.txt" if mode == "train" else "sep_testlist.txt"
        list_path = os.path.join(root_dir, list_filename)
        self.sequence_dirs = []
        if os.path.exists(list_path):
            with open(list_path, "r") as f:
                lines = f.read().splitlines()
            self.sequence_dirs = [os.path.join(root_dir, "sequences", x) for x in lines]
        else:
            raise RuntimeError(f"List file {list_path} does not exist.")

        print(f"Loaded {len(self.sequence_dirs)} sequences from {root_dir} for mode {mode}")

        self.sequence_transform = sequence_transform

        print(f"Loaded {self.__len__()} sequences")

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, idx):
        sequence_dir = self.sequence_dirs[idx]
        crop_trans = None

        # Load the 7 images in the sequence
        rgb_images = []
        yuv_images = []

        for i in range(self.n_frames):
            image_path = os.path.join(sequence_dir, f"im{i+1}.png")
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size

            # Convert to tensor and apply any additional transforms
            rgb_image = self.transform(image)

            # Apply cropping if specified
            if self.crop is not None:
                if crop_trans is None:
                    crop_trans = RandomCropTransform(self.crop[0], self.crop[1], img_width, img_height)
                rgb_image = crop_trans(rgb_image)

            rgb_images.append(rgb_image)

            # Convert to YUV and store
            yuv_image = self.yuv_transform(rgb_image.unsqueeze(0))
            if isinstance(yuv_image, tuple):  # YUV420 case
                yuv_images.append(yuv_image)
            else:  # YUV444 case
                yuv_images.append(yuv_image.squeeze(0))

        # Stack the sequences
        rgb_sequence = torch.stack(rgb_images, dim=0)

        rgb_sequence = torch.stack(rgb_images, dim=0)
        if self.yuv_transform is not None:
            if isinstance(yuv_images[0], tuple):
                y_frames = torch.stack([img[0].squeeze(0) for img in yuv_images], dim=0)
                uv_frames = torch.stack([img[1].squeeze(0) for img in yuv_images], dim=0)
                output = {"rgb": rgb_sequence, "yuv": (y_frames, uv_frames)}
            else:
                yuv_sequence = torch.stack(yuv_images, dim=0)
                output = {"rgb": rgb_sequence, "yuv": yuv_sequence}
        else:
            output = {"rgb": rgb_sequence}

        if self.sequence_transform is not None:
            output = self.sequence_transform(output)

        return output


def generate_train_test_split(
    video_dir,
    train_split=0.8,
    train_filename="sep_trainlist.txt",
    test_filename="sep_testlist.txt",
    seed=42,
):
    """
    Automatically generate training and testing splits for the videos in video_dir.
    It lists all MP4 files, shuffles them (with a fixed seed for reproducibility),
    splits them into train/test sets according to train_split, and saves the filenames
    (without path) to the corresponding text files.

    Args:
        video_dir (str): Directory containing MP4 videos.
        train_split (float): Fraction of files to use for training.
        train_filename (str): Filename for saving training split.
        test_filename (str): Filename for saving test split.
        seed (int): Random seed for shuffling.

    Returns:
        tuple: (list of train filenames, list of test filenames)
    """
    all_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    random.seed(seed)
    random.shuffle(all_files)
    split_index = int(len(all_files) * train_split)
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]

    with open(os.path.join(video_dir, train_filename), "w") as f:
        for name in train_files:
            f.write(name + "\n")
    with open(os.path.join(video_dir, test_filename), "w") as f:
        for name in test_files:
            f.write(name + "\n")

    print(f"Generated split: {len(train_files)} train files, {len(test_files)} test files.")
    return train_files, test_files


def cache_video_frames_info(video_files, output_path):
    """
    Cache video frame counts to avoid reopening videos each time the dataset is created.

    Args:
        video_files (list): List of video file paths to process
        output_path (str): Path to save the cache file

    Returns:
        dict: Mapping of video filenames to frame counts
    """
    video_info = {}

    for path in video_files:
        filename = os.path.basename(path)
        if filename in video_info:
            continue

        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_info[filename] = total_frames

    # Save to disk
    with open(output_path, "w") as f:
        for filename, frames in video_info.items():
            f.write(f"{filename},{frames}\n")

    return video_info


def load_video_frames_info(cache_path):
    """
    Load cached video frame information.

    Args:
        cache_path (str): Path to the cache file

    Returns:
        dict: Mapping of video filenames to frame counts
    """
    video_info = {}

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            for line in f.read().splitlines():
                parts = line.strip().split(",")
                if len(parts) == 2:
                    filename, frames = parts
                    video_info[filename] = int(frames)

    return video_info


class Vimeo90kMP4Dataset(Dataset):
    """
    Dataset for loading full-length Vimeo videos in MP4 format and sampling a contiguous segment
    of frames, generating the same output as the original Vimeo90kDataset.

    Args:
        video_dir (str): Directory containing full-length Vimeo MP4 video files.
        n_frames (int): Number of consecutive frames to sample from each video.
        transform (callable, optional): Transform to apply to each frame (e.g. additional transforms).
        crop (tuple, optional): (crop_height, crop_width) to perform random cropping on each frame.
        yuv_format (str, optional): '444' (default) or '420' to convert frames to YUV.
    """

    def __init__(
        self,
        video_dir,
        n_frames=7,
        transform=None,
        crop=None,
        yuv_format="444",
        sequence_transform=None,
        mode="train",
        generate_split=False,
        train_split=0.8,
        use_cache=True,
    ):
        self.video_dir = video_dir
        self.n_frames = n_frames
        self.transform = transforms.ToTensor()
        if transform is not None:
            self.transform = transforms.Compose([transform, self.transform])

        if isinstance(crop, int):
            crop = (crop, crop)
        self.crop = crop

        self.yuv_transform = RGBtoYUVTransform(yuv_format) if yuv_format else None
        self.sequence_transform = sequence_transform

        all_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
        split_filename = "sep_trainlist.txt" if mode == "train" else "sep_testlist.txt"
        split_path = os.path.join(video_dir, split_filename)

        if not os.path.exists(split_path) and generate_split:
            generate_train_test_split(video_dir, train_split)

        if os.path.exists(split_path):
            with open(split_path, "r") as f:
                split_list = f.read().splitlines()
            video_files = [os.path.join(video_dir, f) for f in all_files if f in split_list]
        else:
            video_files = [os.path.join(video_dir, f) for f in all_files]

        cache_filename = f"video_frames_cache_{mode}.txt"
        cache_path = os.path.join(video_dir, cache_filename)

        if use_cache and os.path.exists(cache_path):
            # Load from cache
            video_info = load_video_frames_info(cache_path)

            self.video_files = []
            self.video_total_frames = []

            for path in video_files:
                filename = os.path.basename(path)
                if filename in video_info and video_info[filename] >= n_frames:
                    self.video_files.append(path)
                    self.video_total_frames.append(video_info[filename])
        else:
            # No cache or not using cache - process videos and optionally create cache
            video_paths_to_process = [os.path.join(video_dir, f) for f in all_files]
            if use_cache:
                video_info = cache_video_frames_info(video_paths_to_process, cache_path)

                self.video_files = []
                self.video_total_frames = []

                for path in video_files:
                    filename = os.path.basename(path)
                    if filename in video_info and video_info[filename] >= n_frames:
                        self.video_files.append(path)
                        self.video_total_frames.append(video_info[filename])
            else:
                # Original implementation
                self.video_files = []
                self.video_total_frames = []
                for path in video_files:
                    cap = cv2.VideoCapture(path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if total_frames >= n_frames:
                        self.video_files.append(path)
                        self.video_total_frames.append(total_frames)
                    else:
                        print(f"skipping video {path}")

        print(f"Found {len(self.video_files)} video files for mode {mode} in {video_dir}")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        total_frames = self.video_total_frames[idx]

        start_frame = random.randint(0, total_frames - self.n_frames)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        rgb_images = []
        yuv_images = []
        crop_trans = None

        for _ in range(self.n_frames):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError(f"Failed to read enough frames from {video_path}.")

            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            img_width, img_height = pil_img.size

            rgb_tensor = self.transform(pil_img)

            if self.crop is not None:
                if crop_trans is None:
                    crop_trans = RandomCropTransform(self.crop[0], self.crop[1], img_width, img_height)
                rgb_tensor = crop_trans(rgb_tensor)

            rgb_images.append(rgb_tensor)

            if self.yuv_transform is not None:
                yuv_tensor = self.yuv_transform(rgb_tensor.unsqueeze(0))
                if isinstance(yuv_tensor, tuple):  # YUV420
                    yuv_images.append(yuv_tensor)
                else:  # YUV444
                    yuv_images.append(yuv_tensor.squeeze(0))

        cap.release()

        rgb_sequence = torch.stack(rgb_images, dim=0).contiguous()
        yuv_sequence = torch.stack(yuv_images, dim=0).contiguous()

        if self.yuv_transform is not None:
            if isinstance(yuv_images[0], tuple):
                y_seq = [pair[0].squeeze(0) for pair in yuv_images]
                uv_seq = [pair[1].squeeze(0) for pair in yuv_images]
                output = {
                    "rgb": rgb_sequence,
                    "yuv": (torch.stack(y_seq, dim=0), torch.stack(uv_seq, dim=0)),
                }
            else:
                output = {
                    "rgb": rgb_sequence,
                    "yuv": yuv_sequence,
                }
        else:
            output = {"rgb": rgb_sequence}

        if self.sequence_transform is not None:
            output = self.sequence_transform(output)

        return output


# ------EXAMPLE-------

# dataset = Vimeo90kDataset("/Users/origovi/Desktop/grp/dataset_task/vimeo_test_clean")
# first = dataset[0]
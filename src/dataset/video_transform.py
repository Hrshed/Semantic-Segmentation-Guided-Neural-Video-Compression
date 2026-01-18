import random
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F


class RandomCropTransform:
    """
    A PyTorch Transform object for randomly cropping an image to a specified resolution.

    Args:
        crop_width (int): Width of the cropped region.
        crop_height (int): Height of the cropped region.
    """

    def __init__(self, crop_width, crop_height, image_width, image_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

        if image_width < self.crop_width or image_height < self.crop_height:
            raise ValueError("Crop size must be smaller than the image dimensions.")

        # Determine the top-left corner of the crop randomly
        self.x_start = random.randint(0, image_width - self.crop_width)
        self.y_start = random.randint(0, image_height - self.crop_height)

    def __call__(self, image):
        """
        Apply the random crop to the input image.

        Args:
            image (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Cropped image.
        """

        # Perform the crop
        cropped_image = F.crop(
            image, self.y_start, self.x_start, self.crop_height, self.crop_width
        )

        return cropped_image


class RGBtoYUVTransform:
    def __init__(self, yuv_format="444"):
        if yuv_format not in ["444", "420"]:
            raise ValueError("yuv_format must be either '444' or '420'")
        self.yuv_format = yuv_format
        # ITU-R BT.709 coefficients
        self.Kr = 0.2126
        self.Kg = 0.7152
        self.Kb = 0.0722

    def __call__(self, rgb):
        """
        Args:
            rgb: Tensor of shape (..., 3, H, W) with values in range [0, 1]
        Returns:
            For YUV444: Tensor of shape (..., 3, H, W)
            For YUV420: Tuple of tensors (y: ..., 1, H, W, uv: ..., 2, H/2, W/2)
        """
        if self.yuv_format == "444":
            return self._rgb_to_yuv444(rgb)
        else:
            return self._rgb_to_yuv420(rgb)

    def _rgb_to_yuv444(self, rgb):
        r, g, b = rgb.chunk(3, dim=-3)

        # Convert to Y
        y = self.Kr * r + self.Kg * g + self.Kb * b

        # Convert to Cb and Cr
        cb = 0.5 * (b - y) / (1 - self.Kb) + 0.5
        cr = 0.5 * (r - y) / (1 - self.Kr) + 0.5

        # Clamp values
        y = torch.clamp(y, 0.0, 1.0)
        cb = torch.clamp(cb, 0.0, 1.0)
        cr = torch.clamp(cr, 0.0, 1.0)

        return torch.cat([y, cb, cr], dim=-3)

    def _rgb_to_yuv420(self, rgb):
        if rgb.size(-2) % 2 != 0 or rgb.size(-1) % 2 != 0:
            raise ValueError("Height and width must be even for YUV420")

        # First convert to YUV444
        r, g, b = rgb.chunk(3, dim=-3)

        # Convert to Y
        y = self.Kr * r + self.Kg * g + self.Kb * b

        # Convert to Cb and Cr
        cb = 0.5 * (b - y) / (1 - self.Kb) + 0.5
        cr = 0.5 * (r - y) / (1 - self.Kr) + 0.5

        # Downsample Cb and Cr
        cb = cb.unfold(-2, 2, 2).unfold(-1, 2, 2)
        cr = cr.unfold(-2, 2, 2).unfold(-1, 2, 2)

        cb = cb.mean(dim=(-2, -1))
        cr = cr.mean(dim=(-2, -1))

        # Clamp values
        y = torch.clamp(y, 0.0, 1.0)
        cb = torch.clamp(cb, 0.0, 1.0)
        cr = torch.clamp(cr, 0.0, 1.0)

        uv = torch.cat([cb, cr], dim=-3)

        return y, uv


class RandomRotationSequence:
    """
    Rotate the entire video sequence by the same random angle within a specified range.

    Args:
        degrees (float): Maximum rotation angle in degrees. The rotation angle is sampled
                         uniformly from [-degrees, degrees].
    """

    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, sample):
        angle = random.uniform(-self.degrees, self.degrees)
        # Rotate all frames in the 'rgb' key
        if "rgb" in sample:
            rgb_seq = sample["rgb"]
            rotated = torch.stack(
                [F.rotate(frame, angle, expand=False) for frame in rgb_seq], dim=0
            )
            sample["rgb"] = rotated
        if "yuv" in sample:
            if isinstance(sample["yuv"], tuple):
                y, uv = sample["yuv"]
                y_rotated = torch.stack(
                    [F.rotate(frame, angle, expand=False) for frame in y], dim=0
                )
                uv_rotated = torch.stack(
                    [F.rotate(frame, angle, expand=False) for frame in uv], dim=0
                )
                sample["yuv"] = (y_rotated, uv_rotated)
            else:
                sample["yuv"] = torch.stack(
                    [F.rotate(frame, angle, expand=False) for frame in sample["yuv"]],
                    dim=0,
                )
        return sample


class RandomHorizontalFlipSequence:
    """
    Randomly horizontally flip the entire video sequence with a given probability.
    This transform applies consistently across all frames in the sequence.

    Args:
        prob (float): Probability of flipping the sequence.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            # Flip RGB sequence if available
            if "rgb" in sample:
                sample["rgb"] = torch.flip(
                    sample["rgb"], dims=[3]
                )  # flip width dimension
            # Flip YUV sequence if available
            if "yuv" in sample:
                if isinstance(sample["yuv"], tuple):
                    y, uv = sample["yuv"]
                    sample["yuv"] = (torch.flip(y, dims=[3]), torch.flip(uv, dims=[3]))
                else:
                    sample["yuv"] = torch.flip(sample["yuv"], dims=[3])
        return sample


class ColorJitterSequence:
    """
    Apply color jitter consistently to both the 'rgb' and 'yuv' sequences.
    The same jitter parameters are applied to all frames across both modalities.

    Args:
        brightness (float or tuple): How much to jitter brightness.
        contrast (float or tuple): How much to jitter contrast.
        saturation (float or tuple): How much to jitter saturation.
        hue (float or tuple): How much to jitter hue.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )


    def __call__(self, sample):
        if "rgb" in sample:
            rgb_seq = sample["rgb"]
            jittered_rgb = torch.stack(
                [self.color_jitter(frame) for frame in rgb_seq], dim=0
            )
            sample["rgb"] = jittered_rgb

        # Since we're only concerned with YUV444, apply to YUV sequence
        if "yuv" in sample and not isinstance(sample["yuv"], tuple):
            yuv_seq = sample["yuv"]
            jittered_yuv = torch.stack(
                [self.color_jitter(frame) for frame in yuv_seq], dim=0
            )
            sample["yuv"] = jittered_yuv

        return sample

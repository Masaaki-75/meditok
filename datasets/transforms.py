import os
import PIL
import torch
import random
import numpy as np

from PIL import Image
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = (1024 * 1024 * 1024 // 4 // 3) * 5
ImageFile.LOAD_TRUNCATED_IMAGES = False

from torchvision.transforms import v2 as transv2



OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)
    

def stable_linear_transform(x, y_min=0, y_max=1, x_min=None, x_max=None, do_clip=True):
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max

    if do_clip:
        x = x.clip(x_min, x_max) if hasattr(x, 'clip') else max(min(x, x_max), x_min)

    x_normalized = (x - x_min) / (x_max - x_min)
    y = y_min + (y_max - y_min) * x_normalized
    return y


class ReadMedicalImage(transv2.Transform):
    _transformed_types = (torch.Tensor, Image.Image, np.ndarray)
    def __init__(self, output_min=-1, output_max=1, modality=None, ct_bias=1024, ct_window_probs=(0.2, 0.3, 0.3, 0.15, 0.05), p=0.5):
        super().__init__()
        self.p = p
        self.ct_window_probs = ct_window_probs
        self.output_min = output_min
        self.output_max = output_max
        self.modality = modality
        self.ct_bias = ct_bias
        self.ct_windows = (
            (-1000, 2000),  # full
            (-1000, 1000),  # common
            (-150, 250),  # soft tissue
            (-1400, 200),  # lung
            (-500, 1300),  # bone
        )
    
    def make_params(self, flat_inputs):
        apply = random.random() < self.p

        if self.modality == 'ct':
            if apply:
                window = np.random.choice(self.ct_windows, p=self.ct_window_probs)
            else:
                window = (-1000, 1000)
        else:
            window = (0, 255)

        return {"window": window}

    def transform(self, inpt, params):

        if isinstance(inpt, np.ndarray):
            img = np.atleast_3d(inpt)
        elif isinstance(inpt, Image.Image):
            if self.modality == 'ct':
                img = np.array(inpt, dtype=np.int16) - self.ct_bias
            else:
                img = np.array(inpt.convert('RGB'))
        else:
            img = inpt

        img = self.to_rgb_tensor(img)

        window = params["window"]
        min_val, max_val = min(window), max(window)
        img = stable_linear_transform(img, y_min=self.output_min, y_max=self.output_max, x_min=min_val, x_max=max_val)
        
        return img
    
    def to_rgb_tensor(self, x):
        x = torch.tensor(x, dtype=torch.float32).squeeze()
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3 and (x.shape[-1] <= 4):  # 有些图像是4通道
            x = x.permute(2, 0, 1).contiguous()
        
        if x.shape[0] == 1:
            x = x.repeat_interleave(3, dim=0)
        elif x.shape[0] > 3:
            x = x[:3]
        
        if x.shape[0] != 3:
            raise ValueError(f"Invalid shape: {x.shape}")
        
        return x
        


class RandomFixedRotation(transv2.Transform):
    def __init__(self, angles=(90, -90), p=0.5):
        super().__init__()
        assert 0. <= p <= 1.
        self.angles = angles
        self.p = p

    def make_params(self, flat_inputs):
        apply = random.random() < self.p
        angle = random.choice(self.angles) if apply else 0
        return {"apply": apply, "angle": angle}

    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        
        angle = params['angle']
        k = angle // 90
        if k > 0:
            return torch.rot90(inpt, k=k, dims=[-2,-1])
        elif k < 0:
            return torch.rot90(inpt, k=k, dims=[-1,-2])
        else:
            return inpt


class RandomGrayScale(transv2.Transform):
    def __init__(self, p=0.2):
        super().__init__()
        assert 0. <= p <= 1.
        self.p = p
        self.transf = transv2.Grayscale(num_output_channels=3)

    def make_params(self, flat_inputs):
        apply = random.random() < self.p
        return {"apply": apply}

    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        else:
            return self.transf(inpt)


class RandomColorJitter(transv2.Transform):
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0., p=0.8):
        super().__init__()
        assert 0. <= p <= 1.
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.transf = transv2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def make_params(self, flat_inputs):
        apply = random.random() < self.p
        return {"apply": apply}
    
    def transform(self, inpt, params):
        if not params["apply"]:
            return inpt
        else:
            return self.transf(inpt)



def build_image_transform(
    image_size: int,
    is_train: bool,
    scale_range=(0.9, 1.0),
    color_jitter_prob=0,
    color_jitter_kwargs=None,
    grayscale_prob=0,
    flip_prob=0,
    rot_kwargs=None,
    rot_prob=0,
):

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]
    
    color_jitter_kwargs = {} if color_jitter_kwargs is None else color_jitter_kwargs
    rot_kwargs = {} if rot_kwargs is None else rot_kwargs

    if is_train:
        transforms = [
            transv2.RandomResizedCrop(scale=scale_range, size=image_size, antialias=True),
            transv2.RandomHorizontalFlip(p=flip_prob),
            transv2.RandomVerticalFlip(p=flip_prob),
            RandomFixedRotation(p=rot_prob, **rot_kwargs),
        ]

        if color_jitter_prob:
            transforms.append(RandomColorJitter(p=color_jitter_prob, **color_jitter_kwargs))

        if grayscale_prob:
            transforms.append(RandomGrayScale(grayscale_prob))
        
    else:
        transforms = [transv2.Resize(image_size, antialias=True)]

    return transv2.Compose(transforms)

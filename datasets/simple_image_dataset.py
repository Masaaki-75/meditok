import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transv2
from datasets.transforms import stable_linear_transform


def read_jsonl(file_path, encoding='utf-8', skip_error=False):
    data = []
    with open(file_path, 'r', encoding=encoding) as f:
        for idx, line in enumerate(f):
            try:
                data.append(json.loads(line.strip()))  # Convert each JSONL line to a dictionary
            except Exception as err:
                print(f"Error when loading Line {idx} in {file_path}: {err}")
                if skip_error:
                    continue
                else:
                    raise err
    return data


class SimpleImageDataset(Dataset):
    def __init__(
        self, 
        meta_path, 
        img_key='identifier', 
        root_dir=None, 
        image_size=(256, 256),
        output_channel=1,  # input channels of the tokenizer, model-specific
        output_min=-1,  # min value of the input, model-specific
        output_max=1,  # max value of the input, model-specific
    ):
        self.images = self.read_meta(meta_path)
        self.is_csv = hasattr(self.images, 'iloc')
        self.img_key = img_key
        self.root_dir = root_dir
        self.image_size = image_size
        self.output_channel = output_channel
        self.output_min = output_min 
        self.output_max = output_max  
        self.transforms = self.get_common_transforms()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        row = self.images.iloc[idx] if self.is_csv else self.images[idx]
        image_path = self.get_image_path(row)
        image = self.read_image(image_path)
        return image, image_path
    
    @staticmethod
    def read_meta(input_path: str):
        if input_path.endswith('.csv'):
            return pd.read_csv(input_path)
        elif input_path.endswith('.jsonl'):
            return read_jsonl(input_path)
        else:
            raise NotImplementedError(f"Currently unsupported data format: {input_path}")
    
    def get_common_transforms(self,):
        return transv2.Resize(size=self.image_size, antialias=True)

    def get_image_path(self, row):
        identifier = row[self.img_key]
        root_dir = None
        if self.root_dir is None:
            root_dir = row.get('root_dir', None)
        else:
            root_dir = self.root_dir
        
        image_path = identifier if root_dir is None else os.path.join(root_dir, identifier)
        return image_path
    
    def read_image(self, image_path):
        image = np.array(Image.open(image_path).convert('RGB'))
        image = self.to_rgb_tensor(image)
        image = self.apply_common_normalization(image)
        return self.transforms(image)
    
    def to_rgb_tensor(self, x):
        x = torch.FloatTensor(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3 and (x.shape[-1] <= 3):
            x = x.permute(2, 0, 1).contiguous()
        
        if x.shape[0] == 1 and (x.shape[0] != self.output_channel):
            x = x.repeat_interleave(self.output_channel, dim=0)
        
        return x
    
    def apply_common_normalization(self, x, window=(0, 255)):
        if window is not None:
            input_min, input_max = min(window), max(window)
        else:
            input_min, input_max = x.min(), x.max()

        x = stable_linear_transform(
            x, 
            input_min=input_min, 
            input_max=input_max,
            output_min=self.output_min, 
            output_max=self.output_max
        )
        
        return x
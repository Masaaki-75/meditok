import os
import io
import re
import sys
import json
import math
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from collections.abc import Mapping


def print_array_info(arr_list, name_list=None, compute_stat=False):
    if not isinstance(arr_list, (list, tuple)):
        arr_list = [arr_list]
    
    name_list = [i for i in range(len(arr_list))] if name_list is None else name_list
    for i in range(len(arr_list)):
        arr = arr_list[i]
        if hasattr(arr, 'min') and hasattr(arr, 'max') and hasattr(arr, 'shape'):
            min_val, max_val = arr.min(), arr.max()
            if hasattr(min_val, 'item'):
                min_val = min_val.item()
            if hasattr(max_val, 'item'):
                max_val = max_val.item()

            val_range = max_val-min_val
            shape = tuple(arr.shape)

            if compute_stat:
                mean_val = arr.mean()
                std_val = arr.std()
                msg = f'INFO {name_list[i]}: shape={shape}, [min,max]=[{min_val:.4f},{max_val:.4f}], range={val_range:.4f}, mean={mean_val:.4f}, std={std_val:.4f}.'
            else:
                msg = f'INFO {name_list[i]}: shape={shape}, [min,max]=[{min_val:.4f},{max_val:.4f}], range={val_range:.4f}.'

        else:
            msg = f'INFO {name_list[i]}: Not an array, type {type(arr)}.'
        
        print(msg)


def stable_linear_transform(x, y_min=0, y_max=1, x_min=None, x_max=None, do_clip=True):
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max

    if do_clip:
        x = x.clip(x_min, x_max) if hasattr(x, 'clip') else max(min(x, x_max), x_min)

    x_normalized = (x - x_min) / (x_max - x_min)
    y = y_min + (y_max - y_min) * x_normalized
    return y


def read_image(img):
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        img = img.convert('RGB')
    return img


def image_to_tensor(x):
    # [H, W, C] -> [B, C, H, W]
    x = torch.FloatTensor(np.array(x)).permute(2, 0, 1)
    x = stable_linear_transform(x, x_min=0, x_max=255, y_min=-1, y_max=1)
    return x.unsqueeze(0)


def tensor_to_image(x):
    # [B, C, H, W] -> [H, W, C]
    x = x.clip(-1, 1).squeeze(0).permute(1, 2, 0)
    x = stable_linear_transform(x, x_min=-1, x_max=1, y_min=0, y_max=255)
    x = x.numpy().astype(np.uint8)
    return Image.fromarray(x)


def _wrap(obj):
    if isinstance(obj, Mapping):
        return DotDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_wrap(v) for v in obj)
    return obj


class DotDict(dict):
    """Dict with attribute access: d.key <==> d['key'] (when key is a valid identifier)."""
    __slots__ = ()

    # attribute -> item
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = _wrap(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    # ensure nested mappings are wrapped on setitem
    def __setitem__(self, key, value):
        super().__setitem__(key, _wrap(value))


if __name__ == '__main__':
    from models.meditok import MedITok

    args = _wrap(dict(
        embed_dim=768,
        num_query=0,
        model='vitamin_large',
        img_size=256,
        drop_path=0,
        vocab_size=32768,
        vocab_width=64,
        vocab_norm=True,
        vq_beta=0.25,
        le=0.0,  # VQ entropy loss weight
        e_temp=0.01,
        num_codebooks=8,
        quant_proj='attn',
        grad_ckpt=True,
        device='cpu'
    ))

    net = MedITok(args)
    ckpt_path = 'weights/meditok/meditok_simple_v1.pth'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(state_dict)
    net = net.eval()

    img_path = 'assets/sample1.png'
    rec_path = img_path.replace('.png', '_rec.png')

    img = read_image(img_path)
    x = image_to_tensor(img)
    with torch.no_grad():
        y = net.img_to_reconstructed_img(x)
        f = net.encode_image(x)

    print_array_info([x, y, f], name_list=['input', 'recon', 'feature'])
    rec = tensor_to_image(y)
    rec.save(rec_path)

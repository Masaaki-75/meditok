import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image 



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


class Class2ImgDatasetCode(Dataset):
    """
    The visual tokenization process is done offline.
    """
    def __init__(self, args):
        self.code_dir = code_dir = args.code_dir
        num_samples = getattr(args, 'num_samples', None)
        self.tokenizer_name = args.tokenizer_name
        img_path_list = read_jsonl(code_dir)
        self.num_samples = num_samples if num_samples is not None else len(img_path_list)
        self.img_path_list = img_path_list[:self.num_samples]

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        item = self.img_path_list[index]
        code_path = item['identifier']
        code = torch.load(code_path)  # [num_codebooks, 16, 16]
        label = item['label']
        return code, label


def build_c2i_code(args):
    return Class2ImgDatasetCode(args)
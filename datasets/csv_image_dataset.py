import os
import gc
import sys
import logging
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets.common import DataInfo

Image.MAX_IMAGE_PIXELS = (1024 * 1024 * 1024 // 4 // 3) * 5
ImageFile.LOAD_TRUNCATED_IMAGES = False

from utilities import config
from datasets.transforms import ReadMedicalImage, build_image_transform


class CsvImageDataset(Dataset):
    def __init__(
        self, 
        input_filename, 
        img_key='identifier', 
        caption_key='modality', 
        tokenizer=None, 
        root_dir=None, 
        do_aug=False,
        prob_flip=0.5,
        prob_rot=0.5,
        prob_grayscale=0.1,
        image_size=(256, 256),
        output_min=-1,
        output_max=1,
        ct_bias=1024,
        num_samples=None
    ):
        logging.debug(f'Loading csv data from {input_filename}.')
        usecols = [img_key, caption_key]
        required_cols = ['modality', 'dataset_name']
        for c in required_cols:
            if c not in usecols:
                usecols.append(c)
        self.images = pd.read_csv(input_filename, usecols=usecols)
        logging.debug(f'Successfully loaded csv data from {input_filename}.')
        self.img_key = img_key
        self.caption_key = caption_key
        self.tokenize = tokenizer
        self.root_dir = root_dir
        self.do_aug = do_aug
        self.prob_flip = prob_flip
        self.prob_rot = prob_rot
        self.prob_grayscale = prob_grayscale
        self.image_size = image_size
        self.output_min = output_min  # [NOTE] 归一化之后的最小值, model-specific
        self.output_max = output_max  # [NOTE] 归一化之后的最大值, model-specific
        self.ct_bias = ct_bias

        if num_samples is not None:
            num_samples = int(num_samples)
            self.images = self.images.iloc[:num_samples]
        self.num_samples = num_samples

        print(f"[CsvImageDataset] Loaded data (n={len(self.images)}, ct_bias={ct_bias})")
        
        self.read_ct_image = ReadMedicalImage(output_min, output_max, modality='ct', ct_bias=ct_bias)
        self.read_common_image = ReadMedicalImage(output_min, output_max, modality=None, ct_bias=0)
        self.transforms = build_image_transform(
            image_size,
            is_train=do_aug,
            flip_prob=prob_flip,
            rot_prob=prob_rot,
            grayscale_prob=prob_grayscale,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        row = self.images.iloc[idx]
        caption = row[self.caption_key]
        if self.caption_key == 'modality':
            caption = f'A {caption} image'

        image = self.read_image(row)
        text = self.tokenize([caption])[0]
        return image, text
    
    def get_image_path(self, row):
        identifier = row[self.img_key]
        root_dir = None
        if self.root_dir is None:
            root_dir = row.get('root_dir', None)
        else:
            root_dir = self.root_dir
        
        image_path = identifier if root_dir is None else os.path.join(root_dir, identifier)
        return image_path
    
    def read_image(self, row):
        image_path = self.get_image_path(row)
        modality = row.get('modality', 'none')
        
        if modality == 'ct' and self.ct_bias != 0:
            image = self.read_ct_image(Image.open(image_path))
        else:
            image = self.read_common_image(Image.open(image_path))
        
        image = self.transforms(image)
        
        return image
    
    def __del__(self):
        if hasattr(self, 'images'):
            del self.images
        gc.collect()


def get_csvimg_dataset(args: config.Args, is_train, epoch=0, iters=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    num_samples = args.train_num_samples if is_train else args.val_num_samples
    root_dir = args.train_root if is_train else args.val_root
    do_aug = True if is_train else False
    assert input_filename
    dataset = CsvImageDataset(
        input_filename,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        tokenizer=tokenizer,
        root_dir=root_dir,
        do_aug=do_aug,
        prob_flip=args.prob_flip,
        prob_rot=args.prob_rot,
        prob_grayscale=args.prob_grayscale,
        image_size=args.img_size,
        ct_bias=args.ct_bias,
        num_samples=num_samples,
    )

    sampler = DistributedSampler(dataset) if is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.local_bs,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=False
    )
    num_samples = len(dataset)
    num_batches = len(dataloader)

    return DataInfo(dataloader, num_samples, num_batches, sampler=sampler)



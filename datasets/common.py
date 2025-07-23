import os
from dataclasses import dataclass
from multiprocessing import Value
from typing import Union, Iterator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utilities import config
from trainers.sampler import DistInfiniteBatchSampler
from datasets.wds_image_dataset import get_wds_dataset
from datasets.csv_image_dataset import get_csvimg_dataset


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value



@dataclass
class DataInfo:
    dataloader: Union[DataLoader, Iterator]
    num_samples: int = 0
    num_batches: int = 0
    sampler: Union[DistributedSampler, DistInfiniteBatchSampler] = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)


def load_data(args: config.Args, epoch=0, iters=0, tokenizer=None):
    data = {}
    if args.train_data:
        if args.dataset_type == "csvimg":
            data["train"] = get_csvimg_dataset(
                args, is_train=True, epoch=epoch, iters=iters, tokenizer=tokenizer)
        elif args.dataset_type == "wds":
            data["train"] = get_wds_dataset(
                args, is_train=True, epoch=epoch, iters=iters, tokenizer=tokenizer)
        else:
            raise NotImplementedError(f"Unsupported training dataset type: {args.dataset_type}")
            
    if args.val_data:
        if args.dataset_type == "csvimg":
            data["val"] = get_csvimg_dataset(
                args, is_train=False, epoch=epoch, iters=iters, tokenizer=tokenizer)
        elif args.dataset_type == "wds":
            data["val"] = get_wds_dataset(
                args, is_train=False, epoch=epoch, iters=iters, tokenizer=tokenizer)
        else:
            raise NotImplementedError(f"Unsupported validation dataset type: {args.dataset_type}")

    return data



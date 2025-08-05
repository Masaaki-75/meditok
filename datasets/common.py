import os
from dataclasses import dataclass
from multiprocessing import Value
from typing import Union, Iterator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utilities import config
from trainers.sampler import DistInfiniteBatchSampler


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






from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
import os
import torch
import numpy as np

DATASET_PATH = 'src/data/datasets/omniglot/Omniglot/'
NUM_TRAIN = 14

def load_dataset(dataset_cfg):
    tasks = sorted(list(filter(lambda x: x[0]!='.', os.listdir(os.path.abspath(DATASET_PATH))))) # filter os.path.isdir
    if dataset_cfg.ALPHABET == '':
        train_datasets, test_datasets = [], []
        for task_idx, subdir in enumerate(tasks):
            train_datasets.append( MultiHeadDataLoader(root = os.path.abspath(DATASET_PATH+subdir), train=True, num_tasks=len(tasks), task_idx=task_idx) )
            test_datasets.append( MultiHeadDataLoader(root = os.path.abspath(DATASET_PATH+subdir), train=False, num_tasks=len(tasks), task_idx=task_idx) )
        trainset = ConcatDataset(train_datasets)
        testset = ConcatDataset(test_datasets)
    else:
        trainset = MultiHeadDataLoader(root = os.path.abspath(DATASET_PATH+dataset_cfg.ALPHABET), train=True, num_tasks=1)
        testset = MultiHeadDataLoader(root = os.path.abspath(DATASET_PATH+dataset_cfg.ALPHABET), train=False, num_tasks=1)

    return (trainset, testset)

class MultiHeadDataLoader(ImageFolder):
    def __init__(
        self,
        root: str,
        train: bool = True,
        num_tasks: int = 1,
        task_idx: int = 1,
    ):
        super().__init__(
            root,
        )
        if train:
            filtered_samples = list(filter(lambda sample: int(sample[0][-6:-4]) <= NUM_TRAIN, self.samples)) #TODO Shuffle
        else:
            filtered_samples = list(filter(lambda sample: int(sample[0][-6:-4]) > NUM_TRAIN, self.samples))
        self.samples = filtered_samples
        self.num_tasks = num_tasks
        self.task_idx = task_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.num_tasks > 1:
            multi_target = np.zeros(self.num_tasks)-1
            multi_target[self.task_idx] = target
            target = multi_target

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

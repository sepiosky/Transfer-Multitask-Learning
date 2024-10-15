from src.data.preprocess_utils.preprocessing import Preprocessor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from yacs.config import CfgNode
from typing import List, Tuple
from .datasets import *
from src import data
import numpy as np
import torch
import matplotlib.pyplot as plt


DATASETS_PATH = 'src.data.datasets'

class ProcessedDataset(Dataset):

    def __init__(self, list_data: List, node_cfg_dataset: CfgNode, is_training: bool):
        self.list_image_data = list_data
        self.size_data = len(list_data)
        self.is_training = is_training
        self.preprocess = node_cfg_dataset.PREPROCESS
        self.preprocessor = Preprocessor(node_cfg_dataset)

    def __len__(self) -> int:
        return self.size_data

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        x, labels = self.list_image_data[index]
        x = np.array(x)
        if self.preprocess:
            x = self.preprocessor(x, self.is_training)
        return x, labels

class DataBatchCollator(object):

    def __init__(self, is_training):
        self.is_training = is_training

    def __call__(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = [x[1] for x in batch]
        if hasattr(labels[0], "__len__"): # labels are list or np array themselves (multi-label)
            labels = np.vstack(labels)
        labels = torch.tensor(labels, dtype=torch.int64)

        inputs = np.array([x[0] for x in batch])
        inputs = torch.tensor(inputs)
        inputs = inputs.permute([0, 3, 1, 2]) # sync with torch nn format (C,W,H)

        return inputs, labels


def build_data_loader(data_list: List, data_cfg: CfgNode, is_training: bool, debug: bool) -> DataLoader:
    dataset = ProcessedDataset(data_list, data_cfg, is_training)
    collator = DataBatchCollator(is_training)
    batch_size = data_cfg.BATCH_SIZE

    num_workers = min(batch_size, data_cfg.CPU_NUM)
    if debug:
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, sampler=SubsetRandomSampler(np.arange(100)), num_workers=num_workers)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training, collate_fn=collator, num_workers=num_workers)
    return data_loader

def load_dataset(dataset_cfg):
    for dataloader_name, dataloader_module in globals().items():
        if dataloader_name == dataset_cfg.NAME:
            return dataloader_module.load_dataset(dataset_cfg)
    print(f"====== Error: No Dataset with Name:{dataset_cfg.NAME} found in Data/Datasets Folder ======")
    exit()

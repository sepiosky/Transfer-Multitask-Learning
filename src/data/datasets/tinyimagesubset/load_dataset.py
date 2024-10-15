from torchvision import datasets
from os import path

def load_dataset(dataset_cfg):
    trainset = datasets.ImageFolder(root=path.abspath('src/data/datasets/tinyimagesubset/train'))
    testset = datasets.ImageFolder(root=path.abspath('src/data/datasets/tinyimagesubset/val'))
    return (trainset, testset)
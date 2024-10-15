from torchvision import datasets
from os import path

def load_dataset(dataset_cfg):
    trainset = datasets.CIFAR10(root=path.dirname(__file__), train=True, download=True)
    testset = datasets.CIFAR10(root=path.dirname(__file__), train=False, download=True)
    return (trainset, testset)
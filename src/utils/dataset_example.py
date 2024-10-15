import argparse
from dataclasses import dataclass
from data import datasets
import data
from src.data.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample Dataset')
    parser.add_argument('--dataset','-d')
    dataset = parser.parse_args().dataset
    classes = 10
    if dataset == 'cifar10':
        train, val = cifar10.load_dataset(None)
        flg = [False for _ in range(classes)]
        sample = []
        while(len(sample) < classes):
            idx = np.random.randint(len(train.data))
            if flg[ train.targets[idx] ] == False:
                flg[ train.targets[idx] ] = True
                sample.append( train.data[idx] )
    if dataset == 'tinyimagesubset':
        train, val = tinyimagesubset.load_dataset(None)
        flg = [False for _ in range(classes)]
        sample = []
        while(len(sample) < classes):
            idx = np.random.randint(len(train.samples))
            if flg[ train.samples[idx][1] ] == False:
                flg[ train.samples[idx][1] ] = True
                sample.append( Image.open(train.samples[idx][0])  )
    fig, ax = plt.subplots(1,10, figsize=(15,5))
    for i in range(classes):
        ax[i].imshow(sample[i])
        ax[i].axes.get_xaxis().set_visible(False)
        ax[i].axes.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.show()
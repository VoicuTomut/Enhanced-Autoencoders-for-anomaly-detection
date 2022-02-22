"""
Utility function for implementing examples.
"""

import torch
from torchvision import datasets, transforms
import numpy as np


def get_dataset(img_width, img_height, train=True, nums=None, limit=None):
    train_set = datasets.MNIST(root='./dataset',
                               train=train,
                               download=True,
                               transform=transforms.Compose(
                                   [transforms.Resize((img_width, img_height)), transforms.ToTensor()])
                               )
    if nums is not None:
        train_set = [image for image in train_set if image[1] in nums]

    if limit is not None:
        return train_set[:limit]

    return train_set

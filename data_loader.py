import os
from glob import glob
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

classes = [
    "back_pack",
    "bike",
    "bike_helmet",
    "bookcase",
    "bottle",
    "calculator",
    "desk_chair",
    "desk_lamp",
    "desktop_computer",
    "file_cabinet",
    "headphones",
    "keyboard",
    "laptop_computer",
    "letter_tray",
    "mobile_phone",
    "monitor",
    "mouse",
    "mug",
    "paper_notebook",
    "pen",
    "phone",
    "printer",
    "projector",
    "punchers",
    "ring_binder",
    "ruler",
    "scissors",
    "speaker",
    "stapler",
    "tape_dispenser",
    "trash_can",
]


class Amazon(torch.utils.data.Dataset):
    def __init__(self, path, source=True, transforms=None, batch_size=32):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        self.source = source
        self.transforms = transforms

    def __len__(self):
        return len(self)

    def __getitem__(self, idx):
        label = 0  # if source (unsupervised training) dummy label

        file = self.files[idx]
        img = Image.open(file)

        if self.source == False:
            label = file.split(self.path)[-1].split("/images/")[-1].split("/")[0]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class Webcam(torch.utils.data.Dataset):
    def __init__(self, path, source=True, transforms=None, batch_size=32):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        self.source = source
        self.transforms = transforms

    def __len__(self):
        return len(self)

    def __getitem__(self, idx):
        label = 0  # if source (unsupervised training) dummy label

        file = self.files[idx]
        img = Image.open(file)

        if self.source == False:
            label = file.split(self.path)[-1].split("/images/")[-1].split("/")[0]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class DSLR(torch.utils.data.Dataset):
    def __init__(self, path, source=True, transforms=None, batch_size=32):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        self.source = source
        self.transforms = transforms

    def __len__(self):
        return len(self)

    def __getitem__(self, idx):
        label = 0  # if source (unsupervised training) dummy label

        file = self.files[idx]
        img = Image.open(file)

        if self.source == False:
            label = file.split(self.path)[-1].split("/images/")[-1].split("/")[0]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

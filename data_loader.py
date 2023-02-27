import os
import random
from glob import glob

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

random.seed(42)
torch.manual_seed(42)

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
    def __init__(self, path, transforms=None, batch_size=32):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(file)

        label = file.split(self.path)[-1].split("/images/")[-1].split("/")[0]
        label = classes.index(label)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class Webcam(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None, batch_size=32):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(file)

        label = file.split(self.path)[-1].split("/images/")[-1].split("/")[0]
        label = classes.index(label)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class DSLR(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None, batch_size=32):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"), recursive=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img = Image.open(file)

        label = file.split(self.path)[-1].split("/images/")[-1].split("/")[0]
        label = classes.index(label)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


# set up the 6 different loaders
root = "/home/gmvincen/class_work/ece_792/Unsupervised-Domain-Adaptation/data"
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
amazon_source = torch.utils.data.DataLoader(
    Amazon(path=os.path.join(root, "amazon"), transforms=transform),
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)
amazon_target = torch.utils.data.DataLoader(
    Amazon(path=os.path.join(root, "amazon"), transforms=transform),
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

webcam_source = torch.utils.data.DataLoader(
    Webcam(path=os.path.join(root, "webcam"), transforms=transform),
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)
webcam_target = torch.utils.data.DataLoader(
    Webcam(path=os.path.join(root, "webcam"), transforms=transform),
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

dslr_source = torch.utils.data.DataLoader(
    DSLR(path=os.path.join(root, "dslr"), transforms=transform),
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)
dslr_target = torch.utils.data.DataLoader(
    DSLR(path=os.path.join(root, "dslr"), transforms=transform),
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

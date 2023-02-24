import os
from glob import glob
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

class Amazon(torch.utils.data.Dataset):
    def __init__(self, path, source=True, transforms=None):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"))
        self.source = source
        self.transforms = transforms
        
    def __len__(self):
        return len(self)
    
    def __getitem__(self, idx):
        label = 0 # if source (unsupervised training) dummy label
        
        file = self.files[idx]
        img = Image.open(file)
        
        if self.source == False:
            label = file.lstrip(self.path).split("/")[0]
        
        if self.transforms is not None:
            img = self.transforms(img)

        
        return img, label
    
class Webcam(torch.utils.data.Dataset):
    def __init__(self, path, source=True, transforms=None):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"))
        self.source = source
        self.transforms = transforms
        
    def __len__(self):
        return len(self)
    
    def __getitem__(self, idx):
        label = 0 # if source (unsupervised training) dummy label
        
        file = self.files[idx]
        img = Image.open(file)
        
        if self.source == False:
            label = file.lstrip(self.path).split("/")[0]
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label
    
class DSLR(torch.utils.data.Dataset):
    def __init__(self, path, source=True, transforms=None):
        self.path = path
        self.files = glob(os.path.join(path, "**", "*.jpg"))
        self.source = source
        self.transforms = transforms
        
    def __len__(self):
        return len(self)
    
    def __getitem__(self, idx):
        label = 0 # if source (unsupervised training) dummy label
        
        file = self.files[idx]
        img = Image.open(file)
        
        if self.source == False:
            label = file.lstrip(self.path).split("/")[0]
        
        if self.transforms is not None:
            img = self.transforms(img)
  
        return img, label
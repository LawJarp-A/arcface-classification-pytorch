import numpy as np

import timm
import math
from fastai.vision.all import *

import torch, torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torchvision.io import read_image

from sklearn import preprocessing

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, df, is_valid=False, transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
        self.basic_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224,224)),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['file_path'].iloc[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.basic_transforms(image)
        label = self.df['class_num'].iloc[idx]-1 # make it start from 0
        label = torch.Tensor([label]).long().squeeze()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return TensorImage(image), label
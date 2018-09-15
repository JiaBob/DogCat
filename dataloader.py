import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob, re

class DogCat(Dataset):
    def __init__(self, path, transform):
        self.img_paths = glob.glob(path)
        self.cls = {'cat': 0, 'dog': 1}
        self.transforms = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_name = self.img_paths[i]
        label = re.search(r'.*\\(.*)\.\d+', img_name).group(1)
        img = Image.open(img_name)
        img = self.transforms(img)
        return img, self.cls[label]
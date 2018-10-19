from torch.utils.data import Dataset
from PIL import Image
import glob, re, os

class DogCat(Dataset):
    def __init__(self, path, transform):
        self.img_paths = glob.glob(path)
        self.cls = {'cat': 0, 'dog': 1}
        self.transforms = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        _, img_name = os.path.split(img_path)  # split file path and file name. dont use re to do it.
        label = re.search(r'(.*)\.\d+', img_name).group(1)  # windows use '/', linux use '\' to split address
        img = Image.open(img_path)
        img = self.transforms(img)
        return img, self.cls[label]
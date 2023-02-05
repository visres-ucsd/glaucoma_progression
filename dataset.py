import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

# input image
# output binary label
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert("RGB") # convert 1 chan to 3 chan
        
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)
        
        return image, label

# input image
# output rnfl thickness map
class RNFLDataset(Dataset):
    def __init__(self, image_paths, label_paths, transforms=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert("RGB") # convert 1 chan to 3 chan
        
        label_path = self.label_paths[idx]
        label = np.load(label_path, allow_pickle=True).astype("float")
        label = np.nan_to_num(label, nan=50.)
        label = torch.from_numpy(label)

        if self.transforms:
            image = self.transforms(image)
        
        return image, label

class ImageRNFLDataset(Dataset):
    def __init__(self, image_paths, rnfl_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.rnfl_paths = rnfl_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert("RGB") # convert 1 chan to 3 chan
        
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)

        rnfl_path = self.rnfl_paths[idx]
        rnfl = np.load(rnfl_path, allow_pickle=True).astype("float")
        rnfl = np.nan_to_num(rnfl, nan=80.)
        rnfl = torch.from_numpy(rnfl).float()

        return (image, rnfl), label

class ImageRNFLVFDataset(Dataset):
    def __init__(self, image_paths, rnfl_paths, vf_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.rnfl_paths = rnfl_paths
        self.vf_paths = vf_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert("RGB") # convert 1 chan to 3 chan
        
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)

        rnfl_path = self.rnfl_paths[idx]
        rnfl = np.load(rnfl_path, allow_pickle=True).astype("float")
        rnfl = np.nan_to_num(rnfl, nan=80.)
        rnfl = torch.from_numpy(rnfl).float()

        vf_path = self.vf_paths[idx]
        vf = np.load(vf_path, allow_pickle=True).astype("float")
        vf = torch.from_numpy(vf).float()

        return (image, rnfl, vf), label

class ImageRNFLVFLongitudinalDataset(Dataset):
    def __init__(self, image_paths, image_paths2, rnfl_paths, rnfl_paths2, vf_paths, vf_paths2, time_deltas, labels, transforms=None):
        self.image_paths = image_paths
        self.rnfl_paths = rnfl_paths
        self.vf_paths = vf_paths
        self.image_paths2 = image_paths2
        self.rnfl_paths2 = rnfl_paths2
        self.vf_paths2 = vf_paths2
        self.time_deltas = time_deltas
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        image = image.convert("RGB") # convert 1 chan to 3 chan

        image_filepath2 = self.image_paths2[idx]
        image2 = Image.open(image_filepath2)
        image2 = image2.convert("RGB") # convert 1 chan to 3 chan

        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)
            image2 = self.transforms(image2)

        rnfl_path = self.rnfl_paths[idx]
        rnfl = np.load(rnfl_path, allow_pickle=True).astype("float")
        rnfl = np.nan_to_num(rnfl, nan=80.)
        rnfl = torch.from_numpy(rnfl).float()

        vf_path = self.vf_paths[idx]
        vf = np.load(vf_path, allow_pickle=True).astype("float")[:156]
        vf = torch.from_numpy(vf).float()

        rnfl_path2 = self.rnfl_paths2[idx]
        rnfl2 = np.load(rnfl_path2, allow_pickle=True).astype("float")
        rnfl2 = np.nan_to_num(rnfl2, nan=80.)
        rnfl2 = torch.from_numpy(rnfl2).float()

        vf_path2 = self.vf_paths2[idx]
        vf2 = np.load(vf_path2, allow_pickle=True).astype("float")[:156]
        vf2 = torch.from_numpy(vf2).float()

        time_delta = self.time_deltas[idx]
        time_delta = np.array([[time_delta]])
        time_delta = torch.from_numpy(time_delta).float()

        return (image, image2, rnfl, rnfl2, vf, vf2, time_delta), label
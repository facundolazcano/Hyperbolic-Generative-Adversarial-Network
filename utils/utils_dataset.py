import sys
import io
import os
import os.path as op
import scipy.io
import numpy as np


import torch
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms, utils, datasets



def loader_CIFAR10(batch_size, path_dataset="/home/jenny2/data/CIFAR10"):
    



    dataset = datasets.CIFAR10(
            path_dataset,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            ),
        )
    
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    
    return loader


def loader_MNIST(batch_size, path_dataset="/home/jenny2/data/MNIST"):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                path_dataset,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
    return loader



class LsunDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path_dataset='/home/jenny2/data/lsun/cat_256', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path_dataset = path_dataset
        self.images = os.listdir(path_dataset)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = np.load(op.join(self.path_dataset, self.images[idx]))

        if self.transform:
            sample = self.transform(sample)

        return sample


def loader_LSUN(batch_size, path_dataset="/home/jenny2/data/lsun/cats_256", distributed=0):
    
    kwargs = {'num_workers': 1, 'pin_memory': True}
    #data.distributed.DistributedSampler(dataset
    
    dataset = LsunDataset(path_dataset,
                          transform=transforms.Compose(
                              [
                                  transforms.ToPILImage(),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ]
                          )
                         )
                                        
    if distributed:
        sampler = data.distributed.DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            **kwargs
        )                                                                     
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )                                                                     
                                      
    return dataloader













class LsunDataset_ant(Dataset):
    """Lmdb dataset."""

    def __init__(self, lmdb_path, resolution=256, transform=None):
        super(LsunDataset, self).__init__()
        import lmdb
        import cv2 
        import PIL.Image
        
        self.env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            self.keys = [key for key, _ in txn.cursor()]

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
            img = img[:, :, ::-1] # BGR => RGB
        
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
        img = np.asarray(img)
        img = img.transpose([2, 0, 1]) # HWC => CHW
        
        if self.transform:
            img = self.transform(img)
            
        return img
        
    def __len__(self):
        return self.length

    

    
    









class TorontoFaceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, TFD_file='TFD_48x48.mat', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        mat = scipy.io.loadmat(TFD_file)
        self.images = mat['images']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.images[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
    

def loader_TFD48(batch_size, path_dataset="/home/jenny2/data/tfd/TFD_48x48.mat"):
    
    kwargs = {'num_workers': 1, 'pin_memory': True}
    
    dataloader = torch.utils.data.DataLoader(
            TorontoFaceDataset(
                path_dataset,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
    return loader

    
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            
            
            
            

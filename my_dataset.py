import scipy.io
import torch
from torch.utils.data import Dataset


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
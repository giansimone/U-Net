import os
from glob import glob
from skimage import io
import torch
from torch.utils.data import Dataset
from senso.utils import im_to_tensor

class SegDataset(Dataset):
    """Dataset to train the U-Net."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # Select the sample
        input_path = self.x[idx]
        target_path = self.y[idx]

        # Load input and target
        x, y = im_to_tensor(input_path).type(torch.float32), torch.from_numpy(io.imread(target_path)).type(torch.long)

        return x, y


def data_set(data_path):
    """Create a dataset to train the U-Net.
    
    Args:
        data_path: Path that contains the data.
    
    Returns:
        dataset: Object from class `SegDataset`.
    """
    inputs = get_files_list(os.path.join(data_path, 'inputs'), 'tif')
    targets = get_files_list(os.path.join(data_path, 'targets'), 'png')
    
    dataset = SegDataset(inputs, targets)
    return dataset


def get_files_list(base_path: str, ext: str):
    """Get a list of files with a specific extension.
    
    Args:
        base_path: Path that contains the files.
        ext: Desired extension for the files.
    Returns:
        filenames: List of filenames with the desired extension.
    """
    filenames = glob(os.path.join(base_path, '*.' + ext))
    filenames.sort()
    return filenames
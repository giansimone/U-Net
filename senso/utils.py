from fileinput import filename
import os
from skimage import io, exposure, img_as_ubyte
from torchvision import transforms
from glob import glob

def im_to_tensor(im_path: str):
    """ Read a TIF image and return a tensor
    
    Args:
        im_path: Path to load the image.
    
    Returns:
        Tensor: Torch tensor containing the image as uint8.
    """
    im = io.imread(im_path)
    im = img_as_ubyte(exposure.rescale_intensity(im))
    im = transforms.ToTensor()(im)
    return im


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
import os
from skimage import io, exposure, img_as_ubyte
from torchvision import transforms

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
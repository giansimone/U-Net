import os
from skimage import io, exposure, img_as_ubyte
from torchvision import transforms

def im_to_tensor(base_path: str, filename: str):
    """ Read a TIF image and return a tensor
    
    Args:
        base_path:
        filename: 
    
    Returns:
        Tensor: Torch tensor containing the image as uin.
    """
    im = io.imread(os.path.join(base_path, filename))
    im = img_as_ubyte(exposure.rescale_intensity(im))
    im = transforms.ToTensor()(im)
    return im
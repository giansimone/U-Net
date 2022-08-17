import torch
import os
from senso.model import UNet
from senso.utils import im_to_tensor


def predict(path_model, base_path, im_filename):
    """Make a prediction using the U-Net."""

    # Get cpu or gpu device for training.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load image on `device`
    im = im_to_tensor(os.path.join(base_path, im_filename))
    im = im.unsqueeze(0)
    im = im.to(device)

    # Build the model and load the weights
    model = UNet().to(device)
    model.load_state_dict(torch.load(os.path.join(path_model, 'model_weights.pth')))
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        pred = model(im)
    
    return pred

if __name__ == '__main__':
    predict('./data', './data/examples', 'sample.tif')
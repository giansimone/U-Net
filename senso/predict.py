import torch
import os
from senso.model import UNet
from senso.utils import im_to_tensor


def predict(path_model: str, im_path: str):
    """Make a prediction using the U-Net.

    Args:
        path_model: Path to the model weights.
        im_path: Path to the image.
    """

    # Get cpu or gpu device for prediction.
    if os.uname().machine == 'arm64':
        # Prediction on Metal
        device = torch.device('mps')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load image on `device`
    im = im_to_tensor(im_path)
    im = im.unsqueeze(0)
    im = im.to(device)

    # Build the model and load the weights
    model = UNet().to(device)
    model.load_state_dict(torch.load(os.path.join(path_model, 'model_weights.pth')))
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        pred = model(im)
    
    return pred.squeeze(0).detach().cpu().numpy()


def mask_from_pred(pred):
    """Calculate a mask from the prediction.
    Args:
        pred: Prediction done for a single image.
    """

    return pred.argmax(0)
import torch
from torch import nn

# Define U-Net model
class UNet(nn.Module):
    """Convolutional neuronal network U-Net.
    """

    def __init__(self) -> None:
        super(UNet, self).__init__()

    
    def forward(self, x):
        pass
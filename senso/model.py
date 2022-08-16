import torch
from torch import nn

# Define a double convolution neuronal network for a U-Net
class ConvConv(nn.Module):
    """Double convolutional neuronal network for a U-Net."""

    def __init__(self, in_ch, out_ch) -> None:
        """
        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
        """
        super(ConvConv, self).__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch)
        )
    
    def forward(self, x):
        x = self.conv_op(x)
        return x


# Define U-Net model
class UNet(nn.Module):
    """Convolutional neuronal network U-Net."""

    def __init__(self, im_height=512, im_weight=512, in_ch=1, out_ch=3) -> None:
        """
        Args:
           im_height: Image height in pixels.
           im_weight: Image with in pixels.
           in_ch: Number of input channels.
           out_ch: Number of output channels (i.e., classes).
        """
        super(UNet, self).__init__()
        self.encode_A = ConvConv(in_ch, 64)
        self.encode_B = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvConv(64, 128)
        )
        self.encode_C = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvConv(128, 256)
        )
        self.encode_D = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvConv(256, 512)
        )
        self.encode_E = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ConvConv(512, 1024)
        )
        self.upsample_A = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decode_A = ConvConv(1024, 512)
        self.upsample_B = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decode_B = ConvConv(512, 256)
        self.upsample_C = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decode_C = ConvConv(256, 128)
        self.upsample_D = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decode_D = ConvConv(128, 64)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, out_ch, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        down_A = self.encode_A(x)
        down_B = self.encode_B(down_A)
        down_C = self.encode_C(down_B)
        down_D = self.encode_D(down_C)
        down_E = self.encode_E(down_D)
        x = self.upsample_A(down_E)
        x = torch.cat([down_D, x], dim=1)
        x = self.decode_A(x)
        x = self.upsample_B(x)
        x = torch.cat([down_C, x], dim=1)
        x = self.decode_B(x)
        x = self.upsample_C(x)
        x = torch.cat([down_B, x], dim=1)
        x = self.decode_C(x)
        x = self.upsample_D(x)
        x = torch.cat([down_A, x], dim=1)
        x = self.decode_D(x)
        x = self.conv_out(x)
        return x
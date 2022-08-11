from model import UNet
from .utils import read_im

# Get cpu or gpu or mps device for training.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

im = read_im()

x.to(device)

model = UNet().to(device)

y = model(x)

with torch.no_grad():
    output = model(input_batch)
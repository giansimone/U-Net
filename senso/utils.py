from turtle import st
from PIL import Image
from torchvision import transforms
import numpy as np

def read_im(filename: str):
    return Image.open(filename)

def im_to_tensor(im):
    return transforms.ToTensor()(im)

## TO DO: Adapt the code below to your framework
m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m, std=s),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model = model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

print(torch.round(output[0]))
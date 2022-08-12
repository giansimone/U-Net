import torch
from torch import nn

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define a stochatic gradiend descent as optimiser
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

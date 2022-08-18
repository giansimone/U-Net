import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from senso.model import UNet
from senso import data


def train(model_path, data_path, epochs=10, batch_size=4, learning_rate=1e-4):
    """Train function for the U-Net.

    Args:
        model_path:
        epochs:
        bactch_size:
        learning_rate:
    """

    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Get cpu or gpu device for training.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps')

    # Load data
    dataset = data.data_set(data_path)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)

    # Build the model
    model = UNet().to(device)

    # Define the loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()
    #optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimiser = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        print(f'Epoch {e+1}\n-------------------------------')
        size = len(data_loader.dataset)

        # Train the model
        model.train()
        for batch, (X, y) in enumerate(data_loader):
            # Load inputs and labels on `device`
            X, y = X.to(device), y.to(device)

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            current_loss, current = loss.item(), batch * len(X)
            print(f'loss: {current_loss:>7f}  [{current:>5d}/{size:>5d}]')

        # Save model weights
        torch.save(model.state_dict(), os.path.join(model_path, 'model_weights.pth'))


if __name__ == '__main__':
    train('./data', './data/examples/cheetah/patches_train/')
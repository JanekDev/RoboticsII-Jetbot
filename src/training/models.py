import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    '''
    538k parameters model
    '''
    def __init__(self, input_dims = (3, 224, 224)) -> None:
        super(SimpleCNN, self).__init__()
        
        in_channels = input_dims[0]
        
        self.sequnce = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3888, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # Print number of parameters
        print(f'Number of parameters: {sum(p.numel() for p in self.parameters())}')
        
    def forward(self, x):
        return self.sequnce(x)

class SimpleCNN2(nn.Module):
    '''
    I don't remember how this model behaves
    '''
    def __init__(self, input_dims = (3, 224, 224)) -> None:
        super(SimpleCNN2, self).__init__()
        
        in_channels = input_dims[0]
        
        self.sequnce = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3200, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # Print number of parameters
        print(f'Number of parameters: {sum(p.numel() for p in self.parameters())}')
        
    def forward(self, x):
        return self.sequnce(x)
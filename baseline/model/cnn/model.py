"""3D CNN model for voxel boxes."""
from torch import nn, tensor
import torch


class CNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        """Initialization.

        Args:
            num_classes: number of amino acids classes.
            in_channels: Channels of input signal.
        """
        super().__init__()
        self.conv_layer1 = nn.Conv3d(in_channels, 100, kernel_size=(3, 3, 3))
        self.conv_layer2 = nn.Conv3d(100, 200, kernel_size=(3, 3, 3))
        self.conv_layer3 = nn.Conv3d(200, 400, kernel_size=(3, 3, 3))
        self.max_pooling = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(400*3**3, num_classes)
        self.softmax = nn.functional.softmax

    def forward(self, x: tensor) -> tensor:
        """Forward function of 3DCNN.

        Args:
            x: input signal.
        """
        x = self.relu(self.conv_layer1(x))
        x = self.relu(self.conv_layer2(x))
        x = self.max_pooling(x)
        x = self.relu(self.conv_layer3(x))
        x = self.max_pooling(x)
        x = self.softmax(self.mlp(x.reshape(x.shape[0], -1)), dim=-1)
        return x


"""3D CNN model for voxel boxes."""
from torch import nn, tensor


class CNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, drop_out: float):
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
        self.mlp1 = nn.Linear(400*3**3, 1000)
        self.mlp2 = nn.Linear(1000, 100)
        self.mlp3 = nn.Linear(100, num_classes)
        self.softmax = nn.functional.softmax
        self.dropout_3d = nn.Dropout3d(p=drop_out)
        self.dropout = nn.Dropout(p=drop_out)
        self.instance_norm0 = nn.InstanceNorm3d(in_channels)
        self.batch_norm1 = nn.BatchNorm3d(100)
        self.batch_norm2 = nn.BatchNorm3d(200)
        self.batch_norm3 = nn.BatchNorm3d(400)

    def forward(self, x: tensor) -> tensor:
        """Forward function of 3DCNN.

        Args:
            x: input signal.
        """
        x = self.instance_norm0(x)

        x = self.dropout_3d(self.relu(self.conv_layer1(x)))
        x = self.batch_norm1(x)

        x = self.dropout_3d(self.relu(self.conv_layer2(x)))
        x = self.max_pooling(x)
        x = self.batch_norm2(x)

        x = self.dropout_3d(self.relu(self.conv_layer3(x)))
        x = self.max_pooling(x)
        x = self.batch_norm3(x)

        x = self.dropout(self.relu(self.mlp1(x.reshape(x.shape[0], -1))))
        x = self.dropout(self.relu(self.mlp2(x)))
        x = self.mlp3(x)
        x = self.softmax(x.reshape(x.shape[0], -1), dim=-1)
        return x

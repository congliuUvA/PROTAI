"""3D CNN model for voxel boxes."""
from torch import nn, tensor
from torchsummary import summary
import torch
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


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Block, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=stride, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm3d(out_channel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=5, stride=stride, bias=False),
            nn.BatchNorm3d(out_channel)
        )

    def forward(self, x):
        print(x.shape)
        out = self.left(x)
        print("leftout", out.shape)
        print("shortcut", self.shortcut(x).shape)
        out = out + self.shortcut(x)
        out = nn.ReLU()(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers: list, num_classes: int = 20, in_channels: int = 4):
        super(ResNet3D, self).__init__()

        self.instance_norm1 = nn.InstanceNorm3d(in_channels)
        self.conv_layer1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3))
        self.batch_norm1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU()

        self.in_channels = 32

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.avg_pool3d = nn.AvgPool3d(2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.instance_norm1(x)
        x = self.conv_layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)  # (32, 18, 18, 18)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  # (512, 2, 2, 2)

        out = self.avg_pool3d(x4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

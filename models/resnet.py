import torch
from torch import nn
from torch.nn import  functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=(2, 1)))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def resnet():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU())
    b2 = nn.Sequential(*resnet_block(64, 64, 2, True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 11))
    net = nn.Sequential(b1, b2, b3, b4, b5, b6)

    return net


if __name__ == '__main__':
    X1 = torch.rand(size=(32, 1, 512, 6))
    X2 = torch.rand(size=(32, 1, 102, 6))

    for layer in resnet():
        X1 = layer(X1)
        print(layer.__class__.__name__, "output shape:\t", X1.shape)

    for layer in resnet():
        X2 = layer(X2)
        print(layer.__class__.__name__, "output shape:\t", X2.shape)
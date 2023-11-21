import torch
from torch import nn
from models import resnet

class DualPathResNet(nn.Module):
    def __init__(self):
        super(DualPathResNet, self).__init__()

        self.b11 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        self.b12 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.ReLU())
        self.b21 = nn.Sequential(*resnet.resnet_block(64, 64, 2, True))
        self.b22 = nn.Sequential(*resnet.resnet_block(64, 64, 2, True))

        self.downsample = nn.Conv2d(64, 64, kernel_size=(7, 1), stride=(5, 1))

        self.b3 = nn.Sequential(*resnet.resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet.resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet.resnet_block(256, 512, 2))
        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 11))

    def forward(self, x):
        x1, x2 = x[0], x[1]
        y1 = self.b11(x1)
        y1 = self.b21(y1)
        y1 = self.downsample(y1)

        y2 = self.b12(x2)
        y2 = self.b22(y2)

        y = torch.cat((y1, y2), dim=-1)
        y = self.b3(y)
        y = self.b4(y)
        y = self.b5(y)
        y = self.b6(y)

        return y


if __name__ == '__main__':
    x1 = torch.rand(size=(32, 1, 512, 6))
    x2 = torch.rand(size=(32, 1, 102, 6))
    net = DualPathResNet()
    y = net((x1, x2))
    print(y.shape)

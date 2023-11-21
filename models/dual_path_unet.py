import torch
from torch import nn
from models import unet

class DualPathUNet(nn.Module):
    def __init__(self, base_c_1, base_c_2):
        super(DualPathUNet, self).__init__()
        self.unet1 = unet.UNet(base_c=base_c_1)
        self.unet2 = unet.UNet(base_c=base_c_2)


    def forward(self, x):
        x1, x2 = x[0], x[1]
        y1, y2 = self.unet1(x1), self.unet2(x2)
        y = (y1, y2)
        return y


if __name__ == '__main__':
    x1 = torch.rand(size=(32, 1, 512, 6))
    x2 = torch.rand(size=(32, 1, 102, 6))
    net = DualPathUNet()
    y1, y2 = net((x1, x2))
    print(y1.shape, y2.shape)

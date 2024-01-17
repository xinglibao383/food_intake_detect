import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool1d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, L]
        diff_length = x2.size()[2] - x1.size()[2]

        # padding_left, padding_right
        x1 = F.pad(x1, [diff_length // 2, diff_length - diff_length // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv1d(in_channels, num_classes, kernel_size=1)
        )


class UNet1D(nn.Module):
    def __init__(self,
                 in_channels: int = 12,
                 num_classes: int = 12,
                 base_c: int = 64):
        super(UNet1D, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)
        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.up1(x5, x4)
        x3 = self.up2(x4, x3)
        x2 = self.up3(x3, x2)
        x1 = self.up4(x2, x1)
        x = self.out_conv(x1)
        return x


net = UNet1D()
x = torch.rand(size=(16, 12, 512))
x_hat = net(x)
# print(x_hat.shape)
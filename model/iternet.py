import torch
import torch.nn as nn
import torch.nn.functional as F


class OneConv(nn.Module):
    """(convolution => ReLU => dropout) * 1"""

    def __init__(self, in_channels, out_channels, kernel_size, do):
        super().__init__()
        self.one_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(do)
        )

    def forward(self, x):
        x = self.one_conv(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => ReLU => dropout) * 2 => MaxPool"""

    def __init__(self, in_channels, out_channels, kernel_size, do):
        super().__init__()
        self.one_conv_1 = OneConv(in_channels, out_channels, kernel_size, do)
        self.one_conv_2 = OneConv(out_channels, out_channels, kernel_size, do)

    def forward(self, x):
        x = self.one_conv_1(x)
        x = self.one_conv_2(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, do, num_classes=1):

        super().__init__()

        self.double_conv_1 = DoubleConv(
            in_channels, out_channels, kernel_size, do)
        self.max_pool_1 = nn.MaxPool2d(2)

        self.double_conv_2 = DoubleConv(
            out_channels, out_channels*2, kernel_size, do)
        self.max_pool_2 = nn.MaxPool2d(2)

        self.double_conv_3 = DoubleConv(
            out_channels*2, out_channels*4, kernel_size, do)
        self.max_pool_3 = nn.MaxPool2d(2)

        self.double_conv_4 = DoubleConv(
            out_channels*4, out_channels*8, kernel_size, do)
        self.max_pool_4 = nn.MaxPool2d(2)

        self.double_conv_5 = DoubleConv(
            out_channels*8, out_channels*16, kernel_size, do)
        self.ConvTranspose2d_5 = nn.ConvTranspose2d(
            out_channels*16, out_channels*8, kernel_size=(2, 2), stride=(2, 2))

        self.double_conv_6 = DoubleConv(
            out_channels*16, out_channels*8, kernel_size, do)
        self.ConvTranspose2d_6 = nn.ConvTranspose2d(
            out_channels*8, out_channels*4, kernel_size=(2, 2), stride=(2, 2))

        self.double_conv_7 = DoubleConv(
            out_channels*8, out_channels*4, kernel_size, do)
        self.ConvTranspose2d_7 = nn.ConvTranspose2d(
            out_channels*4, out_channels*2, kernel_size=(2, 2), stride=(2, 2))

        self.double_conv_8 = DoubleConv(
            out_channels*4, out_channels*2, kernel_size, do)
        self.ConvTranspose2d_8 = nn.ConvTranspose2d(
            out_channels*2, out_channels, kernel_size=(2, 2), stride=(2, 2))

        self.double_conv_9 = DoubleConv(
            out_channels*2, out_channels, kernel_size, do)

        self.double_conv_10 = DoubleConv(
            out_channels, num_classes, kernel_size, do)

    def forward(self, x):

        conv_1 = self.double_conv_1(x)
        pool_1 = self.max_pool_1(conv_1)

        conv_2 = self.double_conv_2(pool_1)
        pool_2 = self.max_pool_2(conv_2)

        conv_3 = self.double_conv_3(pool_2)
        pool_3 = self.max_pool_3(conv_3)

        conv_4 = self.double_conv_4(pool_3)
        pool_4 = self.max_pool_4(conv_4)

        conv_5 = self.double_conv_5(pool_4)
        conv_5 = self.ConvTranspose2d_5(conv_5)

        up_6 = torch.cat([conv_5, conv_4], dim=1)
        conv_6 = self.double_conv_6(up_6)
        conv_6 = self.ConvTranspose2d_6(conv_6)

        up_7 = torch.cat([conv_6, conv_3], dim=1)
        conv_7 = self.double_conv_7(up_7)
        conv_7 = self.ConvTranspose2d_7(conv_7)

        up_8 = torch.cat([conv_7, conv_2], dim=1)
        conv_8 = self.double_conv_8(up_8)
        conv_8 = self.ConvTranspose2d_8(conv_8)

        up_9 = torch.cat([conv_8, conv_1], dim=1)
        conv_9 = self.double_conv_9(up_9)

        out_1 = self.double_conv_10(conv_9)

        return out_1

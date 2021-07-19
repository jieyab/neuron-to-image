import torch
import torch.nn as nn


# UNET paper submodules
class ConvBlock(nn.Module):
    def __init__(self, input_channels, mid_channels, out_channels, BN=False):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, mid_channels, 3, 1, 1)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        self.activation2 = nn.ReLU()
        self.bn = BN
        if (self.bn):
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        if (self.bn):
            x = self.activation1(self.bn1(self.conv1(x)))
            x = self.activation2(self.bn2(self.conv2(x)))
        else:
            x = self.activation1((self.conv1(x)))
            x = self.activation2((self.conv2(x)))
        return x


class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        return self.downsample(x)


class Up(nn.Module):
    def __init__(self, input_channels, output_channels, scale=2):
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv = nn.Conv2d(input_channels, output_channels, 2)

    def forward(self, x):
        x = self.conv(self.pad(self.upsample(x)))
        return x


class Merge(nn.Module):
    def __init__(self):
        super(Merge, self).__init__()

    def forward(self, x1, x2):
        return torch.cat((x1, x2), 1)


# Paper implementation of UNET with a sigmoid activation layer at the end
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, BN=True):
        super(UNet, self).__init__()
        self.block1A = ConvBlock(input_channels, 64, 64, BN)
        self.down1 = Down()
        self.block2A = ConvBlock(64, 128, 128, BN)
        self.down2 = Down()
        self.block3A = ConvBlock(128, 256, 256, BN)
        self.down3 = Down()
        self.block4A = ConvBlock(256, 512, 512, BN)
        self.down4 = Down()
        self.blockMid = ConvBlock(512, 1024, 1024, BN)
        self.up4 = Up(1024, 512)
        self.merge4 = Merge()
        self.block4B = ConvBlock(1024, 512, 512, BN)
        self.up3 = Up(512, 256)
        self.merge3 = Merge()
        self.block3B = ConvBlock(512, 256, 256, BN)
        self.up2 = Up(256, 128)
        self.merge2 = Merge()
        self.block2B = ConvBlock(256, 128, 128, BN)
        self.up1 = Up(128, 64)
        self.merge1 = Merge()
        self.block1B = ConvBlock(128, 64, 64, BN)
        self.finalConv = nn.Conv2d(64, output_channels, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        shorcut1 = self.block1A(x)
        x = self.down1(shorcut1)

        shorcut2 = self.block2A(x)
        x = self.down2(shorcut2)

        shorcut3 = self.block3A(x)
        x = self.down3(shorcut3)

        shorcut4 = self.block4A(x)
        x = self.down4(shorcut4)

        x = self.blockMid(x)

        x = self.up4(x)
        x = self.merge4(x, shorcut4)
        x = self.block4B(x)

        x = self.up3(x)
        x = self.merge3(x, shorcut3)
        x = self.block3B(x)

        x = self.up2(x)
        x = self.merge2(x, shorcut2)
        x = self.block2B(x)

        x = self.up1(x)
        x = self.merge1(x, shorcut1)
        x = self.block1B(x)

        x = self.finalConv(x)
        # x = self.activation(x)

        return x

    def predict(self, x):
        x = self.activation(x)
        return x
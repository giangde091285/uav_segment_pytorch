import torch.nn as nn
from model.block import *
from model.encoder import *

"""
    seg_model
    * u_net 
    * u_net + psp
    * resnet50 + u_net
"""

class U_net(nn.Module):
    def __init__(self, in_channels, classes):
        super(U_net, self).__init__()   # inherit nn.Module class
        self.classes = classes
        self.in_channels = in_channels

        # encoding
        self.Cov = DoubleConv(self.in_channels, 64)
        self.down1 = DownSample(64,128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)

        # decoding
        self.up1 = UpSample(1024,512)
        self.up2 = UpSample(512,256)
        self.up3 = UpSample(256,128)
        self.up4 = UpSample(128,64)

        # head
        self.out = outConv(64, self.classes)

    def forward(self, x):
        
        x1 = self.Cov(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x4, x5)
        x7 = self.up2(x3, x6)
        x8 = self.up3(x2, x7)
        x9 = self.up4(x1, x8)

        x10 = self.out(x9)

        return x10
    

class unet_psp(nn.Module):
    def __init__(self, in_channels, classes):
        super(unet_psp, self).__init__() 
        self.classes = classes
        self.in_channels = in_channels

        # encoding
        self.Cov = DoubleConv(self.in_channels, 64)
        
        self.psp1 = PSPool(64) 
        self.Cov1 = DoubleConv(64, 128)

        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)

        # decoding
        self.up1 = UpSample(1024,512)
        self.up2 = UpSample(512,256)
        self.up3 = UpSample(256,128)
        self.up4 = UpSample(128,64)

        # head
        self.out = outConv(64, self.classes)

    def forward(self, x):
        
        x1 = self.Cov(x)

        x2 = self.psp1(x1) # psp1
        x2 = self.Cov1(x2)

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x4, x5)
        x7 = self.up2(x3, x6)
        x8 = self.up3(x2, x7)
        x9 = self.up4(x1, x8)

        x10 = self.out(x9)
        return x10
    
    
class ResUNet(nn.Module):
    def __init__(self, in_channels, classes, dev):
        super(ResUNet, self).__init__()
        self.classes = classes
        self.in_channels = in_channels
        self.dev = dev

        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)
        self.out = outConv(64, self.classes)

    def forward(self, x):
        
        md = ResNet50Encoder(in_channel=self.in_channels)
        md.to(device=self.dev)

        [x1, x2, x3, x4, x5] = md(x)

        x6 = self.up1(x4, x5)
        x7 = self.up2(x3, x6)
        x8 = self.up3(x2, x7)
        x9 = self.up4(x1, x8)
        x10 = self.out(x9)
        return x10   



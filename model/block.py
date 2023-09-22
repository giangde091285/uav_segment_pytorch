import torch
import torch.nn as nn
import torch.nn.functional as F

# DoubleCov Layer： (conv-> batch_norm -> ReLU)*2
class DoubleConv (nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=3, padding=1, stride=1, bias=False),  # zero padding 
            nn.BatchNorm2d(out_channels),      
            nn.ReLU(inplace=True),   # inplace: use original memory 

            nn.Conv2d(out_channels, out_channels,
                       kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# DownSample Layer: ( maxpool -> DoubleCov Layer)
class DownSample (nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


# UpSample Layer：( Upsample -> cat -> DoubleCov Layer)
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.high_level = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),  
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        )
        self.doubleCov = DoubleConv(in_channels,out_channels)
   
    def forward(self, low_level_x, high_level_x):
        # tensor: N C H W  , make sure other dimensions (H W) are the same
        # hx : (C/2, 2*H , 2*W)
        high_level_x = self.high_level(high_level_x)
        # cat (in C dim)
        cat_x = torch.cat([low_level_x, high_level_x], dim=1)
        # Double Conv
        x = self.doubleCov(cat_x)
        return x


# outConv Layer: 1*1 conv
class outConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.outconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.outconv(x)
    

"""
###################### U-net #######################
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


"""
###################### PSP #######################
"""

class PSPool(nn.Module):
    def __init__(self, in_channel):
        super(PSPool, self).__init__()
        self.in_channel = in_channel
        
        self.pool1 = nn.MaxPool2d(kernel_size=1,stride=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=4,stride=4) 
        self.pool4 = nn.MaxPool2d(kernel_size=8,stride=8)

        self.conv = nn.Conv2d(kernel_size=1,stride=1,
                              in_channels=in_channel,
                              out_channels=in_channel//4)
        
        self.upsample1 = nn.Upsample(mode='bilinear',scale_factor=1,align_corners=True)
        self.upsample2 = nn.Upsample(mode='bilinear',scale_factor=2,align_corners=True)
        self.upsample3 = nn.Upsample(mode='bilinear',scale_factor=4,align_corners=True)
        self.upsample4 = nn.Upsample(mode='bilinear',scale_factor=8,align_corners=True)

        self.DR = nn.Conv2d(kernel_size = 1, stride = 1,
                       in_channels = self.in_channel * 2,
                       out_channels = self.in_channel)
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        x4 = self.pool4(x)

        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x3 = self.conv(x3)
        x4 = self.conv(x4)

        x1 = self.upsample1(x1)
        x2 = self.upsample2(x2)
        x3 = self.upsample3(x3)
        x4 = self.upsample4(x4)

        x5 = torch.cat([x, x1, x2, x3, x4],dim=1)
        
        # DR
        x6 = self.DR(x5)

        x7 = self.pool5(x6)
        
        return x7

"""
################ Resnet50 ################
"""


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Bottleneck, self).__init__()

        factor = 2

        self.conv1 = nn.Conv2d(kernel_size=1, stride=1, in_channels=in_channel, out_channels=out_channel)
        self.conv2 = nn.Conv2d(kernel_size=3, stride=stride, padding=1, in_channels=out_channel, out_channels=out_channel)
        self.conv3 = nn.Conv2d(kernel_size=1, stride=1, in_channels=out_channel, out_channels=out_channel*factor)
        self.conv_x = nn.Conv2d(kernel_size=1, stride=stride, in_channels=in_channel, out_channels=out_channel*factor)

        self.lay_1 = nn.Sequential(self.conv1, nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.lay_2 = nn.Sequential(self.conv2, nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.lay_3 = nn.Sequential(self.conv3, nn.BatchNorm2d(out_channel*factor), nn.ReLU(inplace=True))
        self.lay_x = nn.Sequential(self.conv_x, nn.BatchNorm2d(out_channel*factor))

    def forward(self, x):

        x_add = self.lay_x(x)

        x = self.lay_1(x)
        x = self.lay_2(x)
        x = self.lay_3(x)

        x1 = torch.add(input=x, alpha=1, other=x_add)
        x1 = nn.ReLU(inplace=True)(x1)

        return x1


class ResNet50Encoder(nn.Module):
    def __init__(self, in_channel):
        super(ResNet50Encoder, self).__init__()

         # stage 0
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64,
                      kernel_size=3, stride=1, padding=1),  # 7*7 to 3*3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # (256,256)->(128,128)

        # stage 1
        self.stage1_1 = Bottleneck(in_channel=64, out_channel=64, stride=1)    # 64,64,64,128
        self.stage1_2 = Bottleneck(in_channel=128, out_channel=64, stride=1)   # 128,64,64,128
        self.stage1_3 = Bottleneck(in_channel=128, out_channel=64, stride=1)   # 128,64,64,128

        # stage 2
        self.stage2_1 = Bottleneck(in_channel=128, out_channel=128, stride=2)  # 128,128,128,256 (128,128)->(64,64)
        self.stage2_2 = Bottleneck(in_channel=256, out_channel=128, stride=1)  # 256,128,128,256
        self.stage2_3 = Bottleneck(in_channel=256, out_channel=128, stride=1)  # 256,128,128,256
        self.stage2_4 = Bottleneck(in_channel=256, out_channel=128, stride=1)  # 256,128,128,256

        # stage 3
        self.stage3_1 = Bottleneck(in_channel=256, out_channel=256, stride=2)  # 256,256,256,512 (64,64)->(32,32)
        self.stage3_2 = Bottleneck(in_channel=512, out_channel=256, stride=1)  # 512,256,256,512
        self.stage3_3 = Bottleneck(in_channel=512, out_channel=256, stride=1)  # 512,256,256,512
        self.stage3_4 = Bottleneck(in_channel=512, out_channel=256, stride=1)  # 512,256,256,512
        self.stage3_5 = Bottleneck(in_channel=512, out_channel=256, stride=1)  # 512,256,256,512
        self.stage3_6 = Bottleneck(in_channel=512, out_channel=256, stride=1)  # 512,256,256,512
        self.stage3_7 = Bottleneck(in_channel=512, out_channel=256, stride=1)  # 512,256,256,512

        # stage 4
        self.stage4_1 = Bottleneck(in_channel=512, out_channel=512, stride=2)  # 512,512,512,1024 (32,32)->(16,16)
        self.stage4_2 = Bottleneck(in_channel=1024, out_channel=512, stride=1)  # 1024,512,512,1024
        self.stage4_3 = Bottleneck(in_channel=1024, out_channel=512, stride=1)  # 1024,512,512,1024

    def forward(self, x):

        x = self.stage0(x)
        x1 = x  # (256,256,64)
        
        x = self.pool(x)

        x = self.stage1_1(x)
        x = self.stage1_2(x)
        x = self.stage1_3(x)
        x2 = x  # (128,128,128)

        x = self.stage2_1(x)
        x = self.stage2_2(x)
        x = self.stage2_3(x)
        x = self.stage2_4(x)
        x3 = x  # (64,64,256)

        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage3_3(x)
        x = self.stage3_4(x)
        x = self.stage3_5(x)
        x = self.stage3_6(x)
        x4 = x  # (32,32,512)

        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)
        x5 = x  # (16,16,1024)

        return [x1, x2, x3, x4, x5]

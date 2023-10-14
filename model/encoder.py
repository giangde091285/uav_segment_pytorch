import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    encoder
    * resnet50 
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
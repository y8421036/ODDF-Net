import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision


# def initialize_weights(*models):
#     for model in models:
#         for m in model.modules():
#             if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

class ODNet(nn.Module):
    def __init__(self, in_channels,n_classes):
        super(ODNet, self).__init__()
        self.in_conv1 = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.in_conv2 = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.FPM1 = FPM(4,16,h=16)#128->8 
        self.FPM2 = FPM(16,32,h=4)#8->2
        self.FPM3 = FPM(32,64,h=2)#2->1
        self.fuse1 = Fuse(16)
        self.fuse2 = Fuse(32)
        self.fuse3 = Fuse(64)
        self.disease = classification(64,7)
        self.SegNet2D = UNet(64, 128,n_classes)
        # initialize_weights(self) ###
    def forward(self, x):
        x1 = x[:,0] 
        x2 = x[:,1]
        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x1 = self.in_conv1(x1)
        x2 = self.in_conv2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.FPM1(x)
        x1 = self.fuse1(x)
        x = self.FPM2(x)
        x2 = self.fuse2(x)
        x = self.FPM3(x)
        x3 = self.fuse3(x)
        x = torch.squeeze(x, 2)
        pred2 = self.disease(x)
        x = self.SegNet2D(x,x1,x2,x3)
        # x = self.SegNet2D(x)
        logits = torch.unsqueeze(x, 2)
        return logits, pred2

class classification(nn.Module):
    def __init__(self, channels,classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels*2, channels*4, kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels*2, channels*4, kernel_size=3,padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.max_p = nn.MaxPool2d(4)
        self.ave_p = nn.AvgPool2d(4)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels*8, channels*8, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.outc = nn.Sequential(
            nn.Conv2d(channels*8, classes, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x1 =  self.conv1(self.max_p(x))
        x1 =  self.conv3(self.max_p(x1))
        x1 = self.gap(x1)
        x2 =  self.conv2(self.ave_p(x))
        x2 =  self.conv4(self.ave_p(x2))
        x2 = self.gap(x2)
        x = torch.cat([x1,x2],dim=1)
        x = self.conv5(x)
        x = self.outc(x)
        return x

class Fuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=1)
    def forward(self, x):
        max_out, _ = torch.max(x, dim=2)
        avg_out = torch.mean(x, dim=2)
        x1 = self.conv1(max_out)
        x2 = self.conv2(avg_out)
        x = torch.add(x1,x2)
        x = self.conv2(x)
        return x

class FPM(nn.Module):
    def __init__(self, in_channels,out_channels,h):
        super().__init__()
        self.conv1 =nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(h,1,1), padding=(0,0,0),stride=(h,1,1)),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(h, 3, 3), padding=(0, 1, 1), stride=(h, 1, 1)),
            nn.LeakyReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(h, 3, 3), padding=(0, 2, 2), stride=(h, 1, 1),dilation=(1,2,2)),
            nn.LeakyReLU(inplace=True),
        )
        self.maxpool= nn.MaxPool3d(kernel_size=[h, 1, 1])
        self.conv4 = nn.Sequential(
            nn.Conv3d(out_channels*3+in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        x1=self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4=self.maxpool(x)
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = self.conv4(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels,channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv2D(in_channels, channels)
        self.down = nn.MaxPool2d(2)
        self.down1 = Down(channels,channels)
        self.down2 = Down(channels+16,channels)
        self.down3 = Down(channels+32,channels)
        self.down4 = Down(channels+64,channels)
        # self.down2 = Down(channels,channels)
        # self.down3 = Down(channels,channels)
        # self.down4 = Down(channels,channels)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)
        self.up1 = Up(channels)
        self.up2 = Up(channels)
        self.up3 = Up(channels)
        self.up4 = Up(channels)
        # self.outc = DoubleConv2D(channels, n_classes)
        self.outc = attHead(in_channels, channels)
    def forward(self, x,xx,xxx,xxxx):
    # def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        xx = self.down(xx)
        x22 = torch.cat([x2,xx],dim=1)
        x3 = self.down2(x22)
        # x3 = self.down2(x2)
        x3 = self.drop1(x3)
        xxx = self.down(xxx)
        xxx = self.down(xxx)
        x33 = torch.cat([x3,xxx],dim=1)
        x4 = self.down3(x33)
        # x4 = self.down3(x3)
        x4 = self.drop2(x4)
        xxxx = self.down(xxxx)
        xxxx = self.down(xxxx)
        xxxx = self.down(xxxx)
        x44 = torch.cat([x4,xxxx],dim=1)
        x44 = self.down4(x44)
        # x44 = self.down4(x4)
        x44 = self.drop3(x44)
        x44 = self.up1(x44, x4)
        x44 = self.up2(x44, x3)
        x44 = self.up3(x44, x2)
        feature = self.up4(x44, x1)
        logits = self.outc(x, feature)
        return logits
    
class attHead(nn.Module):
    def __init__(self, channel0, channels):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channel0, channel0, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16,num_channels=channel0),
            nn.LeakyReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 5, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, 5, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, 5, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels, 5, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels, 5, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.head1 = nn.Sequential(
            nn.Conv2d(channel0, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(channel0, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(channel0, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.head4 = nn.Sequential(
            nn.Conv2d(channel0, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.head5 = nn.Sequential(
            nn.Conv2d(channel0, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x0, x):
        x0 = self.conv0(x0)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        att1 = 1 - torch.sigmoid(x1)
        att2 = 1 - torch.sigmoid(x2)
        att3 = 1 - torch.sigmoid(x3)
        att4 = 1 - torch.sigmoid(x4)
        att5 = 1 - torch.sigmoid(x5)
        out1 = torch.mul(x0,att1)
        out2 = torch.mul(x0,att2)
        out3 = torch.mul(x0,att3)
        out4 = torch.mul(x0,att4)
        out5 = torch.mul(x0,att5)
        out1 = self.head1(out1)
        out2 = self.head2(out2)
        out3 = self.head3(out3)
        out4 = self.head4(out4)
        out5 = self.head5(out5)
        out1 = torch.add(x1,out1)
        out2 = torch.add(x2,out2)
        out3 = torch.add(x3,out3)
        out4 = torch.add(x4,out4)
        out5 = torch.add(x5,out5)
        x = torch.cat([out1,out2,out3,out4,out5],dim=1)
        return x
    
class DoubleConv2D(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,  kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(in_channels,out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, channels,  bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(channels // 2, channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(channels*2,channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

import torch
import torch.nn as nn
from torchsummary import summary

class bottle(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(bottle, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        if (in_channels != out_channels):
            self.channelfix = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.channelfix = nn.Identity()
            
    def forward(self, x0):
        x = x0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        x0 = self.channelfix(x0)
        x = x + x0
        x = self.relu3(x)
        return x

class bottleTrans(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(bottleTrans, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        if (in_channels != out_channels):
            self.channelfix = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.channelfix = nn.Identity()
            
    def forward(self, x0):
        x = x0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        x0 = self.channelfix(x0)
        x = x + x0
        x = self.relu3(x)
        return x
    
        
class botnet(nn.Module):
    def __init__(self):
        super(botnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.multibottle1 = nn.Sequential(
            bottle(64, 64, 256),
            bottle(256, 64, 256),
            bottle(256, 64, 256)
        )
        self.multibottle2 = nn.Sequential(
            bottle(256, 128, 512),
            bottle(512, 128, 512),
            bottle(512, 128, 512),
            bottle(512, 128, 512)
        )
        self.multibottle3 = nn.Sequential(
            bottle(512, 256, 1024),
            bottle(1024, 256, 1024),
            bottle(1024, 256, 1024),
            bottle(1024, 256, 1024),
            bottle(1024, 64, 1024),
            bottle(1024, 64, 1024)
        )
        self.multibottle4 = nn.Sequential(
            bottleTrans(1024, 512, 2048),
            bottleTrans(2048, 512, 2048),
            bottleTrans(2048, 512, 2048)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)
    
    def forward(self,x0):
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        x = self.multibottle1(x)
        x = self.multibottle2(x)
        x = self.multibottle3(x)
        x = self.multibottle4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x

mynet = botnet()

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False), # 如果这里的stride不一样，或者outputchannel不一样，那么就需要对shortcut进行处理
            nn.BatchNorm2d(outchannel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False), # padding=1, stride=1的时候输入输出尺寸一致
            nn.BatchNorm2d(outchannel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False), # 采用和前面一样的stride和outputchannel，保证尺寸一致
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.layer(x)
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, ResBlock, num_classes=20):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.make_layer(ResBlock, 32, 32, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 32, 64, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 64, 128, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 128, 128, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
    
    def make_layer(self, block, inchannel, outchannel, num_blocks, stride):
        layers = []
        layers.append(block(inchannel, outchannel, stride))
        for stride in range(num_blocks - 1): # 一个用于变化尺寸，剩下的用于保持尺寸不变
            layers.append(block(outchannel, outchannel, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.functional.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
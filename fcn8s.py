#import fcn
import torch.nn as nn
import numpy as np
import torch

import os.path as osp

# a simple version
# from github.com/knn1989/FCN8s/


class VGGBlock2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class VGGBlock3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out



class FCN8(nn.Module):

    def __init__(self, num_classes, in_channels = 3):
        super(FCN8, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2, ceil_mode = True)

        self.conv_block_1 = VGGBlock2(in_channels, 64, 64)

        self.conv_block_2 = VGGBlock2(64, 128, 128)

        self.conv_block_3 = VGGBlock3(128, 256, 256)

        self.conv_block_4 = VGGBlock3(256, 512, 512)

        self.conv_block_5 = VGGBlock3(512, 512, 512)

        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding = 3),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace = True),
            nn.Dropout2d()
            )

        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace = True),
            nn.Dropout2d()
            )
        
        self.score_fr = nn.Conv2d(4096, num_classes, 1)

        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride = 2, padding = 1, bias = False)

        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride = 8, padding = 4,bias = False)

        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride = 2, padding = 1, bias = False)

    def forward(self, x):
        h = x
        h = self.conv_block_1(h)
        h = self.pool(h) # 1/2

        h = self.conv_block_2(h)
        h = self.pool(h) # 1/4

        h = self.conv_block_3(h)
        h = self.pool(h) # 1/8
        pool3 = h

        h = self.conv_block_4(h)
        h = self.pool(h) # 1/16
        pool4 = h

        h = self.conv_block_5(h)
        h = self.pool(h) # 1/32
        
        h = self.fc6(h) 

        h = self.fc7(h)

        h = self.score_fr(h)

        h = self.upscore2(h)
        upscore2 = h # 1/16

        h = self.score_pool4(pool4)
        score_pool4 = h # 1/16

        h = upscore2 + score_pool4 # 1/16

        h = self.upscore_pool4(h)
        upscore_pool4 = h # 1/8

        h = self.score_pool3(pool3)
        score_pool3 = h # 1/8

        h = upscore_pool4 + score_pool3

        h = self.upscore8(h)

        return h


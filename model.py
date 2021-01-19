
from __future__ import print_function
import numpy as np
from os.path import exists, join
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
from math import sqrt
import sys


class Model1(nn.Module):
    def __init__(self, size):
        super(Model1, self).__init__()
        """
        private parameters
        including the structure of the network
        """
        self.size = size

        self.__conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.__conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.__bn1_1 = nn.BatchNorm2d(64)
        self.__bn1_2 = nn.BatchNorm2d(64)
        self.__conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.__conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.__bn2_1 = nn.BatchNorm2d(128)
        self.__bn2_2 = nn.BatchNorm2d(128)
        self.__conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.__conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.__upconv4 = nn.Conv2d(256, 128, 1)
        self.__bn4 = nn.BatchNorm2d(128)
        self.__bn4_1 = nn.BatchNorm2d(256)
        self.__bn4_2 = nn.BatchNorm2d(256)
        self.__bn4_out = nn.BatchNorm2d(256)
        self.__conv7_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.__conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.__upconv7 = nn.Conv2d(128, 64, 1)
        self.__bn7 = nn.BatchNorm2d(64)
        self.__bn7_1 = nn.BatchNorm2d(128)
        self.__bn7_2 = nn.BatchNorm2d(128)
        self.__bn7_out = nn.BatchNorm2d(128)
        self.__conv9_1 = nn.Conv2d(64, 32, 3, padding=1)
        self.__conv9_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.__bn9_1 = nn.BatchNorm2d(32)
        self.__bn9_2 = nn.BatchNorm2d(32)
        self.__conv9_3 = nn.Conv2d(32, 1, 1)
        
        self.__fc = nn.Linear((size/4)*(size/4),2)

        self.__maxpool = nn.MaxPool2d(
            2, stride=2, return_indices=False, ceil_mode=False)
        self.__upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()

    def forward(self, x1):

        x1 = F.relu(self.__bn1_2(self.__conv1_2(
            F.relu(self.__bn1_1(self.__conv1_1(x1))))))
        x2 = F.relu(self.__bn2_2(self.__conv2_2(
            F.relu(self.__bn2_1(self.__conv2_1(self.__maxpool(x1)))))))
        xup = F.relu(self.__bn4_2(self.__conv4_2(
            F.relu(self.__bn4_1(self.__conv4_1(self.__maxpool(x2)))))))
        xup = self.__bn4(self.__upconv4(self.__upsample(xup)))
        xup = self.__bn4_out(torch.cat((x2, xup), 1))
        xup = F.relu(self.__bn7_2(self.__conv7_2(
            F.relu(self.__bn7_1(self.__conv7_1(xup))))))
        xup = self.__bn7(self.__upconv7(self.__maxpool(xup)))
        
        xup = F.relu(self.__bn9_3(self.__conv9_3(F.relu(self.__bn9_2(
            self.__conv9_2(F.relu(self.__bn9_1(self.__conv9_1(xup)))))))))

        return F.sigmoid(self.__fc(xup))

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

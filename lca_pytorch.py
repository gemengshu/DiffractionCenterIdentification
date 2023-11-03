import numpy as np

import torch
import torch.nn as nn
import torchvision

def slice(x,a,b,c,d):
    return x[:,:,a:b,c:d]

def circ_shift(cen, shift):
    _, _, hei, wid = cen.shape

    ######## B1 #########
    # old: AD  =>  new: CB  
    #      BC  =>       DA
    B1_NW = slice(cen,shift,None,shift,None)
    B1_NE = slice(cen,shift,None,None,shift)
    B1_SW = slice(cen,None,shift,shift,None)
    B1_SE = slice(cen,None,shift,None,shift)
    B1_N = torch.cat([B1_NW, B1_NE], dim=3)
    B1_S = torch.cat([B1_SW, B1_SE], dim=3)
    B1 = torch.cat([B1_N, B1_S], dim=2)

    ######## B2 #########
    # old: A  =>  new: B
    #      B  =>       A
    B2_N = slice(cen,shift,None,None,None)
    B2_S = slice(cen,None,shift,None,None)
    B2 = torch.cat([B2_N, B2_S], dim=2)

    ######## B3 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B3_NW = slice(cen,shift,None,wid-shift,None)
    B3_NE = slice(cen,shift,None,None,wid-shift)
    B3_SW = slice(cen,None,shift,wid-shift,None)
    B3_SE = slice(cen,None,shift,None,wid-shift)
    B3_N = torch.cat([B3_NW, B3_NE], dim=3)
    B3_S = torch.cat([B3_SW, B3_SE], dim=3)
    B3 = torch.cat([B3_N, B3_S], dim=2)

    ######## B4 #########
    # old: AB  =>  new: BA
    B4_W = slice(cen,None,None,wid-shift,None)
    B4_E = slice(cen,None,None,None,wid-shift)
    B4 = torch.cat([B4_W, B4_E], dim=3)

    ######## B5 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B5_NW = slice(cen,hei-shift,None,wid-shift,None)
    B5_NE = slice(cen,hei-shift,None,None,wid-shift)
    B5_SW = slice(cen,None,hei-shift,wid-shift,None)
    B5_SE = slice(cen,None,hei-shift,None,wid-shift)
    B5_N = torch.cat([B5_NW, B5_NE], dim=3)
    B5_S = torch.cat([B5_SW, B5_SE], dim=3)
    B5 = torch.cat([B5_N, B5_S], dim=2)

    ######## B6 #########
    # old: A  =>  new: B
    #      B  =>       A
    B6_N = slice(cen,hei-shift,None,None,None)
    B6_S = slice(cen,None,hei-shift,None,None)
    B6 = torch.cat([B6_N, B6_S], dim=2)

    ######## B7 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B7_NW = slice(cen,hei-shift,None,shift,None)
    B7_NE = slice(cen,hei-shift,None,None,shift)
    B7_SW = slice(cen,None,hei-shift,shift,None)
    B7_SE = slice(cen,None,hei-shift,None,shift)
    B7_N = torch.cat([B7_NW, B7_NE], dim=3)
    B7_S = torch.cat([B7_SW, B7_SE], dim=3)
    B7 = torch.cat([B7_N, B7_S], dim=2)

    ######## B8 #########
    # old: AB  =>  new: BA
    B8_W = slice(cen,None,None,shift,None)
    B8_E = slice(cen,None,None,None,shift)
    B8 = torch.cat([B8_W, B8_E], dim=3)

    s1 = (B1-cen)*(B5-cen)
    s2 = (B2-cen)*(B6-cen)
    s3 = (B3-cen)*(B7-cen)
    s4 = (B4-cen)*(B8-cen)

    c1234 = torch.minimum(s1,torch.minimum(s2,torch.minimum(s3,s4)))
    return c1234

def mlc(cen,d):
    res = []
    for i in range(d):
        res.append(circ_shift(cen,i))
    return torch.maximum(res)

class Blam_weight(nn.Module):
    def __init__(self, in_channels, reduction = 4):
        super(Blam_weight, self).__init__()
        self.reduction = reduction
        self.in_channels = in_channels
        self.mid_channels = in_channels//self.reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.mid_channels, 
                               kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(self.mid_channels, track_running_stats = False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.mid_channels, 
                               self.mid_channels * self.reduction,
                               kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(self.mid_channels * self.reduction, track_running_stats = False)
        self.sigmoid = nn.Sigmoid()
    
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return x 
    

class LCA_Unet(nn.Module):
    def __init__(self, in_channels):
        super(LCA_Unet, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 8*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(8*2, 8*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.blam1 = Blam_weight(8*2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8*2, 16*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(16*2, 16*2, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(16*2, track_running_stats = False),
            nn.ReLU()
        )
        self.blam2 = Blam_weight(16*2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16*2, 32*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(32*2, 32*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.blam3 = Blam_weight(32*2)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(32*2, 64*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(64*2, 64*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32*2, kernel_size = 4, padding = 1, stride = 2),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(64*2, 32*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(32*2, 32*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 16*2, kernel_size = 4, padding = 1, stride = 2),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(32*2, 16*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(16*2, 16*2, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(16*2, track_running_stats = False),
            nn.ReLU()
        )

        self.up9 = nn.Sequential(
            nn.ConvTranspose2d(16*2, 8*2, kernel_size = 4, padding = 1, stride = 2),
            nn.ReLU()
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(16*2, 8*2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(8*2, 8, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(8, 2, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU()
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size = 1, padding = 0, stride = 1),
            nn.Sigmoid()
        )
        
        

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        mlc1 = circ_shift(conv1, 3)
        blam1 = self.blam1(mlc1)

        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        mlc2 = circ_shift(conv2, 3)
        blam2 = self.blam2(mlc2)

        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        mlc3 = circ_shift(conv3, 3)
        blam3 = self.blam3(mlc3)

        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)

        up7 = self.up7(conv4)
        up7 = blam3*up7 + mlc3
        merge7 = torch.cat([conv3, up7], dim = 1)

        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        up8 = blam2*up8 + mlc2
        merge8 = torch.cat([conv2, up8], dim = 1)

        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        up9 = blam1*up9 + mlc1
        merge9 = torch.cat([conv1, up9], dim = 1)

        conv9 = self.conv9(merge9)
        output = self.conv10(conv9)
        return output
    

def initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
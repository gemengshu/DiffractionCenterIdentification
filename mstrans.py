import torch
from torch import nn
import torch.nn.functional as F
import math
from config import *

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel = 3, stride = 1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size = kernel, stride = stride,  padding=1)
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

class VGGBlock_v2(nn.Module):
    def __init__(self, in_channels, stride=1, dilation = 1, bias = False):

        super().__init__()
        middle_channels = in_channels
        out_channels = in_channels

        self.stride = stride

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding = 1, bias = bias)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, stride = stride,
                    dilation = dilation, padding = dilation, bias = bias)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, kernel_size = 1, bias = bias)
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

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        #self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        #self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        #x4_0 = self.conv4_0(self.pool(x3_0))

        #x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

class AxialAttention_dynamic_mod(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic_mod, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values 
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=True)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)
        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)/((self.group_planes//2)**0.5*3)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialBlock_dynamic_mod(nn.Module):
    expansion = 2

    def __init__(self,inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_dynamic_mod, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_dynamic_mod(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_dynamic_mod(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class MS_Trans(nn.Module):
    def __init__(self, input_channels, out_channels, img_size = SIZE):
        super(MS_Trans, self).__init__()


        self.base_width = 64
        self.groups = 8

        dense_layers = [1,1,1]
        att_layers = [2,2,2]
        att_planes = 32
        base_filters = att_planes

        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace = True),
            nn.Conv2d(base_filters, base_filters*2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace = True)
        ) # img_size //2
        self.conv1 = self._make_layer(VGGBlock_v2, base_filters*2, dense_layers[0], stride = 2) # img_size // 4
        self.conv2 = self._make_layer(VGGBlock_v2, base_filters*2, dense_layers[1], stride = 2) # img_size // 8
        self.conv3 = self._make_layer(VGGBlock_v2, base_filters*2, dense_layers[2], stride = 1, dilation=2) # img_size //8

        self.att1 = self._make_att(AxialBlock_dynamic_mod, base_filters*2, att_planes, att_layers[0], kernel_size=img_size//4)
        self.att2 = self._make_att(AxialBlock_dynamic_mod, base_filters*2, att_planes, att_layers[1], kernel_size = img_size//8)
        self.att3 = self._make_att(AxialBlock_dynamic_mod, base_filters*2, att_planes, att_layers[2], kernel_size = img_size//8)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.dec2 = nn.Sequential(
            nn.BatchNorm2d(att_planes*2),
            nn.ReLU(inplace = True),
            nn.Conv2d(att_planes*2, att_planes, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(att_planes),
            nn.ReLU(inplace = True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.dec3 = nn.Sequential(
            nn.BatchNorm2d(att_planes*2),
            nn.ReLU(inplace = True),
            nn.Conv2d(att_planes*2, att_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(att_planes),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(att_planes*2),
            nn.ReLU(inplace = True),
            nn.Conv2d(att_planes*2, att_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(att_planes),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(att_planes*3, att_planes, kernel_size = 1),
            nn.BatchNorm2d(att_planes),
            nn.ReLU(inplace = True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),#img_size//4    
            nn.Conv2d(att_planes, att_planes, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(att_planes),
            nn.ReLU(inplace = True),
            nn.Conv2d(att_planes, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),#img_size//2
            nn.Conv2d(3, 1, kernel_size=1),
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),#img_size
        )


    def _make_layer(self, block, channels, blocks, stride = 1, dilation = 1):
        layers = []
        layers.append(block(channels, stride, dilation=max(1, dilation/2),bias = True))
        for i in range(1, blocks):
            layers.append(block(channels, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_att(self, block, inplanes, planes, blocks, kernel_size=56, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, norm_layer=norm_layer, kernel_size=kernel_size))
        inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv_in(x)
        x1 = self.conv1(x)
        x1_att = self.dec1(self.att1(x1)+x1)
        x2 = self.conv2(x1)
        x2_att = self.dec2(self.att2(x2)+x2)
        x3 = self.conv3(x2)
        x3_att = self.dec3(self.att3(x3)+x3)

        x = torch.cat([x1_att,x2_att,x3_att], dim=1)

        out = self.fusion(x)
        return out

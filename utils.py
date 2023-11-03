from __future__ import print_function

import sys
from os.path import exists, join
from os import listdir
import os
from PIL import Image
import numpy as np

import random

from openpyxl import Workbook
from openpyxl import load_workbook
from config import *

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, RandomRotation
import torchvision.transforms as transforms
from torchvision.transforms import functional


#from sklearn.metrics import jaccard_score

# def calc_iou(input, target):
#     if USE_CUDA:
#         input = input.cpu().numpy().reshape(-1)
#         target = target.cpu().numpy().reshape(-1)
#     else:
#         input = input.numpy().reshape(-1)
#         target = target.numpy().reshape(-1)
#     input = np.where(input>0.5,1,0).astype('uint8')
#     target = target.astype('uint8')
#     return jaccard_score(target, input)

def gen_mask(image, center, r):
    h,w = image.shape
    mask = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            if np.sqrt((i-center[1])**2+(j-center[0])**2)<=r:
                mask[i,j] = 1
    return mask

def norm_tensor(x):
    if len(x.size())==4:
        b,_,_,_ = x.size()
        for i in range(b):
            x[i,:,:,:] = (x[i,:,:,:]-torch.min(x[i,:,:,:]))/(torch.max(x[i,:,:,:])-torch.min(x[i,:,:,:]))
    else:
        if len(x.size())==3:
            b,_,_ = x.size()
            for i in range(b):
                x[i,:,:] = (x[i,:,:]-torch.min(x[i,:,:]))/(torch.max(x[i,:,:])-torch.min(x[i,:,:]))
    return x

def read_centers(file_name, sheet_name = 'Sheet1'):

    wb = load_workbook(file_name)
    sheet = wb[sheet_name]

    row_number = 0
    names_and_centers = []

    for row in sheet.values:
        if row_number == 0:
            row_number = row_number + 1
            continue
        name = row[0]
        x = float(row[1])
        y = float(row[2])
        r = float(row[3])
        names_and_centers.append((name, x, y, r))
        
        row_number = row_number + 1

    return names_and_centers

def cal_center(image, threshold = 0.5):

    h,w = image.shape
    center_h = 0
    center_w = 0
    count = 0
    mini_val = image.min()
    for i in range(h):
        for j in range(w):
            if image[i,j] > threshold:
                center_h += i
                center_w += j
                count += 1
    if count == 0:
        return 0,0
    center_h = center_h/count
    center_w = center_w/count
    return center_w, center_h

def cal_weighted_center(image):
    H,W = image.shape

    h_index_matrix = torch.arange(0,H).view(-1,1).expand(-1,W)
    w_index_matrix = torch.arange(0,W).view(1,-1).expand(H,-1)
    h_index_matrix = h_index_matrix.float()
    w_index_matrix = w_index_matrix.float()
    
    weights = torch.exp(ALPHA*image)
    #weights = image

    out_center_h = torch.sum(weights*h_index_matrix)
    out_center_w = torch.sum(weights*w_index_matrix)
    spatial_scaled = torch.sum(weights)

    out_center_h = torch.div(out_center_h, spatial_scaled)
    out_center_w = torch.div(out_center_w, spatial_scaled)
    
    return out_center_w, out_center_h


def read_data(image_dir, size, centers_file, gray_scale):
    centers_and_names = read_centers(centers_file)

    images = []
    masks = []
    norm_centers = []

    for i in range(len(centers_and_names)):
        image_name = centers_and_names[i][0]
        if gray_scale:
            img = Image.open(image_dir + image_name).convert('L')
        else:
            img = Image.open(image_dir + image_name).convert('RGB')
        target = np.zeros((img.height,img.width))
        center_x, center_y, center_r = centers_and_names[i][1:4]
        
       
        #if not self.mask:
        for i in range(img.height):
            for j in range(img.width):
                if ((i-center_y)**2+(j-center_x)**2) <= center_r**2:
                    target[i,j] = 255
                else:
                    target[i,j] = 0
        
        target = Image.fromarray(target).convert('L')

        if img.width != img.height:
            height = img.height
            width = img.width
            if width > height:

                img = img.crop((int(width-height),0,width, height))
                target = target.crop((int(width-height),0,width, height))
                center_x = center_x - (width-height)
            else:
                img = img.crop((0, int(height-width), width, height))
                target = target.crop((0, int(height-width), width, height))
                center_y = center_y - (height-width)
        norm_center_x = center_x/img.width
        norm_center_y = center_y/img.height
        norm_center_r = center_r/img.width

        
        img = img.resize((size, size))
        target = target.resize((size, size))

        images.append(img)
        masks.append(target)
        norm_centers.append((norm_center_x,norm_center_y,norm_center_r))

    return images, masks, norm_centers

class DatasetFromFolder(data.Dataset):

    def __init__(self, image_dir, size, centers_file, crop_size = 200, gray_scale = True, resize = True, rotate = True, add_noise = False):
        super(DatasetFromFolder, self).__init__()
        self.resize = resize
        self.gray_scale = gray_scale
        self.image_dir = image_dir
        self.size = size
        self.centers_file = centers_file
        self.centers_and_names = read_centers(centers_file)
        
        self.crop_size = crop_size
    
        self.p = 0.5
        self.rotate = rotate

        self.degree = 180

        self.add_noise = add_noise
        self.var = 0.006

        self.images, self.masks, self.norm_centers = read_data(image_dir, size, centers_file, gray_scale)

        return

    def __getitem__(self, index):

        input = self.images[index]
        target = self.masks[index]

        norm_center_x,norm_center_y,_ = self.norm_centers[index]

        transform = ToTensor()
        # augmentation

        if self.crop_size > 0:
            i,j = self.random_crop((input.height, input.width))

            cropped_center = np.array(((norm_center_x*self.size-j)/self.crop_size, (norm_center_y*self.size-i)/self.crop_size))
 
            cropped_input = transforms.functional.crop(input, i, j, self.crop_size, self.crop_size)

            cropped_input = transforms.functional.resize(cropped_input, self.size)
            cropped_target = transforms.functional.crop(target, i, j, self.crop_size, self.crop_size)
            cropped_target = transforms.functional.resize(cropped_target, self.size)

        else:
            cropped_center = np.array((norm_center_x, norm_center_y))
            cropped_input = input
            cropped_target = target

        if self.rotate:
            cropped_input,cropped_target = self.random_rotate(cropped_input, cropped_target, cropped_center)
        if self.add_noise:
            cropped_input, _ = self.random_noise(cropped_input)

        cropped_target = transform(cropped_target)
        cropped_input = transform(cropped_input)
        cropped_input = norm_tensor(cropped_input)

        return cropped_input, cropped_target, cropped_center

    def random_noise(self, image):
        
        array = np.copy(np.array(image)/255.)
        mean = 0
        var = self.var * np.random.rand(1)
        sigma = var ** 0.5

        gaussian = np.random.normal(mean, sigma, (image.height, image.width))

        noisy = np.zeros(array.shape, np.float32)

        if len(array.shape) == 2:
            noisy = array + gaussian
        else:
            noisy[:,:,0] = array[:,:,0] + gaussian
            noisy[:,:,1] = array[:,:,1] + gaussian
            noisy[:,:,2] = array[:,:,2] + gaussian

        noisy = (noisy - np.min(noisy))/(np.max(noisy) - np.min(noisy))

        return Image.fromarray(np.uint8(noisy * 255.)), sigma

    def random_crop(self, ori_size):
        h,w = ori_size
        
        i = torch.randint(0, h-self.crop_size, size = (1, )).item()
        j = torch.randint(0, w-self.crop_size, size = (1, )).item()
        return i,j
    
    def random_rotate(self, input, target, center, mask = None):
        if random.random() > self.p:
            angle = random.randint(0, self.degree)
            input = functional.rotate(input, angle, center = (int(center[0]*self.size+0.5),int(center[1]*self.size+0.5)))
            target = functional.rotate(target, angle,center = (int(center[0]*self.size+0.5),int(center[1]*self.size+0.5)))

        return input, target

    def __len__(self):
        return len(self.centers_and_names)


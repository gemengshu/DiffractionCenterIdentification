from __future__ import print_function
from openpyxl import Workbook
from openpyxl import load_workbook


import numpy as np
from os.path import exists, join
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
from os import listdir
import os
from PIL import Image
import sys

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
        names_and_centers.append((name, x, y))
        
        row_number = row_number + 1

    return names_and_centers


class DatasetFromFolder(data.Dataset):

    def __init__(self, image_dir, size, centers_file, resize = True):

        super(DatasetFromFolder, self).__init__()
        self.resize = resize
        self.image_dir = image_dir
        self.size = size
        self.centers_file = centers_file
        self.centers_and_names = read_centers(centers_file)

    def load_img(self, filepath):

        img = Image.open(filepath).convert('L')
        return img

    def __getitem__(self, index):

        image_name = self.image_dir + self.centers_and_names[index][0]
        input = self.load_img(image_name)

        center = np.array((self.centers_and_names[index][1]/input.width, self.centers_and_names[index][2]/input.height))
        #temp = np.fromiter(iter(input.getdata()), np.uint8)
        if self.resize:
            input = input.resize((self.size, self.size))        

            
        transform = ToTensor()
        input = transform(input)
        center = torch.tensor(torch.tensor(center).float())
        return input, center

    def __len__(self):

        return len(self.centers_and_names)

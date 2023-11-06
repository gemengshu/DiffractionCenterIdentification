
import os

from os.path import exists, join
import cv2 as cv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.autograd import Variable

import torch.nn.functional as F

from utils import DatasetFromFolder
from config import *

from mstrans import MS_Trans

import random


def train_mstrans():
    train_folder = 'training_new/'

    save_folder = 'save_path/'

    test_image = Image.open('test_image.jpg').convert('RGB') 
    test_image = test_image.resize((SIZE, SIZE))
    transform = ToTensor()
    test_image = transform(test_image)
    test_image = torch.unsqueeze(test_image,0)

    test_image_display = cv.cvtColor(np.uint8(np.array(test_image[0,:,:,:].permute(1,2,0).data.numpy()*255.)), cv.COLOR_BGR2GRAY)
    test_image_display = transform(np.array(test_image_display)/255)
    test_image_display = torch.unsqueeze(test_image_display,0)

    model = MS_Trans(3,1)

    pretrain = False

    pretrain_num = 4999
    if pretrain:
        start_epoch = pretrain_num + 1
        epochs = 10000
        lr = 1e-5
    else:
        start_epoch = 0
        epochs = 5000
        lr = 1e-4
    pretrain_path = save_folder + '/model_epoch_{}.pth'.format(pretrain_num)

    if USE_CUDA:
        model = model.cuda()
        if pretrain:
            model.load_state_dict(torch.load(pretrain_path))
    else:
        if pretrain:
            model.load_state_dict(torch.load(pretrain_path, map_location = lambda storage, loc:storage))

    center_file = 'training_new/centers_new.xlsx'

    train_data = DatasetFromFolder(train_folder, SIZE, center_file, crop_size = 480, gray_scale = False, rotate = True, add_noise = False)

    train_data_loader = DataLoader(dataset = train_data, num_workers = THREAD, batch_size = BATCH_SIZE, pin_memory = True, prefetch_factor = 4, shuffle = True)

    optimizer = optim.Adam(
        model.parameters(), lr=lr, betas=(BETA, 0.999), weight_decay=0.0001)

    criterion = nn.BCELoss()


    if USE_CUDA:
        test_image = test_image.cuda()
        criterion = criterion.cuda()

    test_image = (test_image-.5)*2

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        seg_loss = 0
        center_loss = 0
        count = 0
        model.train()
        for iteration, (image, target, center) in enumerate(train_data_loader):
            image = Variable(image)
            target = Variable(target)
            image = (image-.5)*2
            if USE_CUDA:
                image = image.cuda()
                target = target.cuda()
            
            optimizer.zero_grad()
            output = model(image)
            output = torch.sigmoid(output)
            loss = criterion(output, target)

            epoch_loss += loss.cpu().data.numpy()

            loss.backward()
            optimizer.step()

            count += 1
        print("Epoch {}: {}".format(epoch, epoch_loss/count))

        if epoch%100 == 0:
            model.eval()
            test_output = model(test_image)
            test_output = torch.sigmoid(test_output)
            test_output = test_output.cpu().data.double()
            display_image = torch.cat((test_image_display, test_output), dim = 3)
            display_image = np.array(display_image[0,0,:,:].cpu().data)
            display_image = np.uint8(display_image*255)

            cv.imwrite(save_folder + str(epoch) + '.bmp', display_image)
            torch.save(model.state_dict(), save_folder + "model_epoch_{}.pth".format(epoch))

    model_out_path = (save_folder + "model_epoch_{}.pth").format(epoch)
    torch.save(model.state_dict(), model_out_path)
        

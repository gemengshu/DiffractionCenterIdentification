from model import Model1
from utils import DatasetFromFolder
import os
from os.path import exists, join
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import *
import cv2 as cv
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

train_folder = 'data/training/'
save_folder = 'results/'

test_image = Image.open('data/testing/1.bmp').convert('L') 
test_image = test_image.resize((SIZE, SIZE))
transform = ToTensor()
test_image = transform(test_image)
test_image = torch.unsqueeze(test_image,0)

model = Model1(SIZE)

center_file = 'data/training/centers.xlsx'
train_data = DatasetFromFolder(train_folder, SIZE, center_file)
train_data_loader = DataLoader(dataset = train_data, num_workers = THREAD, batch_size = BATCH_SIZE, shuffle = True)

optimizer = optim.Adam(
    model.parameters(), lr=LR, betas=(BETA, 0.999), weight_decay=0.0001)

criterion = nn.SmoothL1Loss()

if USE_CUDA:
    model = model.cuda()
    test_image = test_image.cuda()

for epoch in range(EPOCHS):
    epoch_loss = 0
    count = 0
    model.train()
    for iteration, (image, center) in enumerate(train_data_loader):
        image = Variable(image)
        #center = Variable(center)
        if USE_CUDA:
            image = image.cuda()
            center = center.cuda()
        
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, center)
        epoch_loss += loss.cpu().data.numpy()
        loss.backward()
        optimizer.step()

        count += 1
     
    print("Epoch {}: {}".format(epoch, epoch_loss/count))
    if epoch%50 == 0:
        model.eval()
        test_output = model(test_image)
        display_image = np.array(test_image[0,:,:,:].cpu().data).reshape((SIZE,SIZE))
        display_image = cv.cvtColor(np.uint8(display_image*255), cv.COLOR_GRAY2BGR)
        #gd_center = center.cpu().data
        out_center = test_output.cpu().data
        #cv.circle(display_image, (int(gd_center[0,0]*SIZE), int(gd_center[0,1]*SIZE)), 2, (255,0,0),1)
        cv.circle(display_image, (int(out_center[0,0]*SIZE), int(out_center[0,1]*SIZE)), 2, (0, 0, 255), 1)
        cv.imwrite(save_folder + str(epoch) + '.bmp', display_image)

model_out_path = (save_folder + "/model_epoch_{}.pth").format(epoch)
torch.save(model.state_dict(), model_out_path)
    

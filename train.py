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

train_folder = 'D:/GMS/Documents/Dong/diffractionDL/data/training/'

model = Model1(SIZE)

center_file = 'D:/GMS/Documents/Dong/diffractionDL/data/training/centers.xlsx'
train_data = DatasetFromFolder(train_folder, SIZE, center_file)
train_data_loader = DataLoader(dataset = train_data, num_workers = THREAD, batch_size = BATCH_SIZE, shuffle = True)

optimizer = optim.Adam(
    model.parameters(), lr=LR, betas=(BETA, 0.999), weight_decay=0.0001)

criterion = nn.SmoothL1Loss()

if USE_CUDA:
    model = model.cuda()


for epoch in range(EPOCHS):
    epoch_loss = 0
    count = 0
    for iteration, (image, center) in enumerate(train_data_loader):
        image = Variable(image)
        center = Variable(center)
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

    
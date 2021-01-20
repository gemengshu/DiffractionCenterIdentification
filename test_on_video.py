from model import Model1
import torch
import torch.nn as nn
from torch.autograd import Variable
from config import *
import numpy as np
from torchvision.transforms import ToTensor

import cv2 as cv
from PIL import Image
## saving setting
save_path = 'data/YiSu/results_using_model/'


## load model
model_path = 'results/model_epoch_499.pth'
model = Model1(SIZE)
if USE_CUDA:
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location = lambda storage, loc: storage))

## load video
video_path = 'data/YiSu/In-situDP.avi'
cap  = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*'XVID')

images = []
success, frame = cap.read()
while success:
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    images.append(frame)
    success, frame = cap.read()
cap.release()


## test on frames
if USE_CUDA:
    model = model.cuda()

transform = ToTensor()
model.eval()
centers = []
for i in range(len(images)):
    h,w = images[i].shape
    frame = Image.fromarray(images[i])
    frame = frame.resize((SIZE, SIZE))
    frame_tensor = transform(frame)
    frame_tensor = torch.unsqueeze(frame_tensor, 0)
    if USE_CUDA:
        frame_tensor = frame_tensor.cuda()
    center = model(frame_tensor)
    center = center.cpu().data
    c_w = center[0,0]*w
    c_h = center[0,1]*h
    display_image = cv.cvtColor(images[i], cv.COLOR_GRAY2BGR)
    cv.circle(display_image, (int(c_w), int(c_h)), 2, (0,0,255),1)
    cv.imwrite(save_path + '{}.bmp'.format(i), display_image)
    centers.append((c_w, c_h))

np.save(save_path + 'centers.npy', centers)

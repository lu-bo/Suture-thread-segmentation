import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
import re
import sys
import logging
import os
from albumentations import (
    Compose,
    Normalize,
    Resize,
    PadIfNeeded,
    VerticalFlip,
    HorizontalFlip,
    RandomCrop,
    CenterCrop,
)
from utils import ramps, losses
import ignite.engine as engine
import ignite.handlers as handlers
import ignite.contrib.handlers as c_handlers
import ignite.metrics as imetrics
# visualization
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import (
    OutputHandler,
    OptimizerParamsHandler,
    WeightsScalarHandler,
)
import torchvision.utils as tvutils
import torch.nn.functional as F
# modules
from Models.plane_model import UNet11_DO
# from loss import LossMulti, LossMulti_contour
from dataset import SutureSegDataset, TwoStreamBatchSampler
from metrics import (
    MetricRecord,
    calculate_confusion_matrix_from_arrays,
    calculate_iou,
    calculate_dice,
)
import random

import PIL.Image as pil
from torchvision import transforms
import time
import matplotlib.pyplot as plt

net = UNet11_DO(in_channels=3, num_classes=2, bn=True, has_dropout=False)
# net = model.cuda()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')
net.to(device=device)
# net.load_state_dict(torch.load('/home/trs-learning/Documents/dvrk/dvrk_learning/Pytorch-UNet-master/SutureSeg/trained_model/Proposed/iter_3500.pth', map_location=device))
net.load_state_dict(torch.load('./trained_model/Proposed/iter_3500.pth', map_location=device))
net.eval()
logging.info("Model loaded !")

import cv2
from PIL import Image

#%%
video_path = '../Experiments_2020_Trans/experiment_trans/porcine_pork/Exp15/'
cap = cv2.VideoCapture(video_path + 'ECM_video.avi')
ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path + 'output.mp4', fourcc, 20.0, (640, 480))
i_img = 1

while (ret):
    # frame = frame[0: 480, 640 * 2: 640 * 3]
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # out.write(frame[0 : 480, 0 : 640 * 2])
    input_image = Image.fromarray(frame)
    input_image_pytorch = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image_pytorch = input_image_pytorch.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # tool segmentation
        mask_outputs = net(input_image_pytorch)

        probs = torch.sigmoid(mask_outputs)
        probs = probs.squeeze(0)
        full_mask = probs.squeeze().cpu().numpy()
        suture_mask = (full_mask[1, :, :] > 0.35).astype(np.uint8)

    # suture_mask_cv2 = np.array(suture_mask)
    # out.write(suture_mask * 255)

    # suture_mask_img = cv2.threshold(suture_mask, 127, 255, cv2.THRESH_BINARY)[1] # ensure binary
    # num_labels, labels_im = cv2.connectedComponents(suture_mask_img, connectivity=8)

    # label_hue = np.uint8(179 * labels_im / np.max(labels_im))
    # blank_ch = 255 * np.ones_like(label_hue)
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    #
    # # cvt to BGR for display
    # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    #
    # # set bg label to black
    # labeled_img[label_hue == 0] = 0

    # cv2.imshow('labeled.png', suture_mask_img)

    frame[suture_mask != 0] = (200, 0, 0)

    # cv2.imwrite(video_path + str(i_img) + '_mask.jpg', suture_mask * 255)
    # cv2.imwrite(video_path + str(i_img) + '.jpg', frame)
    i_img += 1

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
# %%


# input_image = Image.fromarray(input_image)
#

#
# count = 0
# while success:
#     # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#     # input_image = pil.open(image_path).convert('RGB')
#
#     input_image = input_image.resize((640, 480))
#     input_image_pytorch = transforms.ToTensor()(input_image).unsqueeze(0)
#     input_image_pytorch = input_image_pytorch.to(device=device, dtype=torch.float32)
#
#     with torch.no_grad():
#         # tool segmentation
#         mask_outputs = net(input_image_pytorch)
#
#         probs = torch.sigmoid(mask_outputs)
#         probs = probs.squeeze(0)
#         full_mask = probs.squeeze().cpu().numpy()
#         suture_mask = (full_mask[1, :, :] > 0.5).astype(np.uint8)
#
#     input_image = np.array(input_image)
#
#     #     suture_mask_3c[:, :, 0] = suture_mask
#     #     suture_mask_3c[:, :, 1] = suture_mask
#     #     suture_mask_3c[:, :, 2] = suture_mask
#
#     #     two_images = cv2.hconcat([input_image, suture_mask])
#     cv2.imshow('frame', input_image)
#     # out.write(suture_mask)
#     #
#     input_image = vidcap.read()
#     #     print('Read a new frame: ', success)
#     # count += 1
#     input_image = input_image[0: 480, 640 * 2 : 640 * 3]
#     input_image = Image.fromarray(input_image)
#
#
# cv2.waitKey(0)

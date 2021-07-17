import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os
import cv2

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, save
import torchvision.ops as ops
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data import random_split
from torch.autograd import Variable
from PIL import Image

from utils.model import *
from utils.dataset import *
from utils.loss import *
from utils.test_utils import *
import time
from constraint_network import *
from tqdm import tqdm

folder = sys.argv[1]

device = torch.device('cpu')

multiplier_inp = int(sys.argv[2])
dropout_rate_inp = float(sys.argv[3])


def DC(pred_mask, gt_mask):
    """(W, H, ch) 0/1 values"""
    pred = np.transpose(pred_mask, (2,1,0))
    gt = np.transpose(gt_mask, (2,1,0))
    # print(pred.shape, gt.shape)
    smooth = 1.
    intersection = np.sum(np.abs(pred*gt), axis=(0,1))
    union = np.sum(gt,(0,1))+np.sum(pred,(0,1))
    dice = (2*intersection+smooth)/(union+smooth)
    precision = (intersection + smooth) / (np.sum(pred, (0,1)) + smooth)
    recall = (intersection + smooth) / (np.sum(gt, (0,1)) + smooth)
    return dice, precision, recall

names = os.listdir("/home/ajey/Desktop/temp/anns/")
names = [''.join(name.split(".")[:-1]) for name in names]
data = Full_Dataset("/home/ajey/Desktop/temp/images/", "/home/ajey/Desktop/temp/anns/", names, transform = transforms.Compose([ToTensor()]))
loader = DataLoader(dataset = data, batch_size = 1, shuffle=True, num_workers=1)

model = unet(3, 3, 5, 5, 7, 7, multiplier_inp, dropout_rate_inp)

model.load_state_dict(torch.load("model_weights/checkpoints", map_location=torch.device('cpu')))

# with torch.no_grad():
#     clinical = LCC()
#     clinical.load_state_dict(torch.load("./model_weights.pt"))
#     print("Loaded constraint network!")

DOWNSIZE = 4
SCALING_FACTOR = 64
length = 0
dice_c = []
dice_v = []
dice_e = []

precision_c = []
precision_v = []
precision_e = []

recall_c = []
recall_v = []
recall_e = []
for counter in range(5, 10):
    FOLDER = folder + "/results" + str(counter)

    if "results" + str(counter) not in os.listdir(folder):
        os.mkdir(FOLDER)
        os.mkdir(FOLDER + "/images")
        os.mkdir(FOLDER + "/groundtruths")
        os.mkdir(FOLDER + "/detections")
    model.eval()

    with torch.no_grad():

        for i, data in tqdm(enumerate(loader), total=len(loader)):
            if i % 1 == 0:
                
                image, mask, box, ebox, vbox, cbox, fname = data["image"], data['mask'], data['inter_box'], data['ebox'], data['vbox'], data['cbox'], data['fname']
                image, mask, box, ebox, vbox, cbox  = image.to(device), mask.to(device), box.to(device), ebox.to(device), vbox.to(device), cbox.to(device) 
                ftrue = open(FOLDER + "/groundtruths/" + fname[0].split("images/")[1].split('.jpg')[0] + ".txt", "w")
                fpred = open(FOLDER + "/detections/" + fname[0].split("images/")[1].split('.jpg')[0] + ".txt", "w")
                pred_seg, pred_box, pred_ebox, pred_vbox, pred_cbox = model(image)
                pred_box[:, 0, :, :] = torch.sigmoid(pred_box[:, 0, :, :])
                ones = torch.ones((1, 3, 480, 640))
                zeros = torch.zeros((1,3,480,640))
                pred_seg = torch.where(pred_seg>0.5, ones, zeros)
                print(torch.unique(mask))
                pred = np.squeeze(pred_seg.numpy(), 0)
                gt = np.squeeze(mask.numpy(), 0)
                D, P, R = DC(pred, gt)
                dice_c.append(D[1])
                dice_v.append(D[0])
                dice_e.append(D[2])

                precision_c.append(P[1])
                precision_v.append(P[0])
                precision_e.append(P[2])

                recall_c.append(R[1])
                recall_v.append(R[0])
                recall_e.append(R[2])
        
                
                ## JUST TO CHECK CONSTRAINT NETWORK ##
                # HVmask = pred_seg[:,0,:,:].view(pred_seg.size(0),307200)
                # HCmask = pred_seg[:,1,:,:].view(pred_seg.size(0),307200)
                # MMmask = pred_seg[:,2,:,:].view(pred_seg.size(0),307200)
                # BBoxes = pred_box.view(pred_box.size(0), 6000)

                # context = torch.cat((HVmask, HCmask, MMmask, BBoxes), dim=1)
                # Ct = clinical(context)
                # Ct = Ct.view(Ct.size(0),30,40)
                # pred_box[:,0,:,:] = Ct
                ## JUST TO CHECK CONSTRAINT NETWORK ##
                
                t_pred_array, t_true_array = convert_to_numpy(pred_box), convert_to_numpy(box)

                prd = t_pred_array[0]
                tru = t_true_array[0]

                im_true = box_image(tru, "true", ftrue, DOWNSIZE, SCALING_FACTOR, 0.5)
                
                npred = run_non_max_suppression(prd, DOWNSIZE, SCALING_FACTOR, float(counter / 10), 0.5, 0.5)
                im_npred = box_image(npred, "pred", fpred, DOWNSIZE, SCALING_FACTOR, float(counter / 10))

                # ##NEW BLOCK##
                # pred_cbox_array, true_cbox_array = convert_to_numpy_pred(pred_cbox), convert_to_numpy(cbox)
                # prd = pred_cbox_array[0]
                # tru = true_cbox_array[0]
                # im_true = box_image(tru, "true", ftrue, DOWNSIZE, SCALING_FACTOR, 0.5)
                # npred = run_non_max_suppression(prd, DOWNSIZE, SCALING_FACTOR, float(counter / 10), 0.5, 0.5)
                # im_npred = box_image(npred, "pred", fpred, DOWNSIZE, SCALING_FACTOR, float(counter / 10))
                # ##NEW BLOCK##
                
                fig, ax = plt.subplots(1, 3, figsize = (15, 5))
                
                ax[0].imshow((im_npred + 255 * convert_torch_to_numpy(pred_seg)[0]).astype(np.uint8))
                ax[0].set_xlabel("Prediction", fontsize=20)
                ax[1].imshow((im_true + 255 * convert_torch_to_numpy(mask)[0]).astype(np.uint8))
                ax[1].set_xlabel("Ground truth", fontsize=20)
                ax[2].imshow(np.transpose((image.cpu().detach().numpy()), (0, 2, 3, 1))[0].astype(np.uint8))
                ax[2].set_xlabel("Real image", fontsize=20)
                plt.savefig(FOLDER + "/images/" + fname[0].split("images/")[1])
                # plt.close()
        dice_c = np.mean(dice_c)
        precision_c = np.mean(precision_c)
        recall_c = np.mean(recall_c)

        dice_v = np.mean(dice_v)
        precision_v = np.mean(precision_v)
        recall_v= np.mean(recall_v)

        dice_e = np.mean(dice_e)
        precision_e = np.mean(precision_e)
        recall_e = np.mean(recall_e)

        print(dice_c, precision_c, recall_c)
        print(dice_v, precision_v, recall_v)
        print(dice_e, precision_e, recall_e)
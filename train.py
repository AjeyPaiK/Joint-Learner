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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data import random_split
from torch.autograd import Variable

from PIL import Image

from utils.model import *
from utils.dataset import *
from utils.loss import *
from utils.test_utils import *
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

folder = sys.argv[1]

total_epochs_inp = int(sys.argv[2])
change_epoch_inp = int(sys.argv[3])

multiplier_inp = int(sys.argv[4])
dropout_rate_inp = float(sys.argv[5])

f = open(folder + "/training_log.txt", "w")

f.write("Total Epochs: {}, Rate Increase Epoch {}, Multiplier: {}, Dropout_Rate: {}\n".format(total_epochs_inp, change_epoch_inp, multiplier_inp, dropout_rate_inp))
# f.write("GPU Information: {}\n".format(torch.cuda.get_device_capability()))


names = os.listdir("Data/anns/")
names = [''.join(name.split(".")[:-1]) for name in names]

train_names, val_names = train_test_split(names, test_size = 0.1)


train_data = Full_Dataset("Data/images", "Data/anns/", train_names, augment = 0.95, flip = 0.95, angles = 0.95, image_transform = transforms.Compose([transforms.ColorJitter(.2, .2, .1, .05)]), transform = transforms.Compose([ToTensor()]))
val_data = Full_Dataset("Data/images", "Data/anns/", val_names, transform = transforms.Compose([ToTensor()]))

train_loader = DataLoader(dataset = train_data, batch_size = 1, shuffle=True, num_workers=16)

val_loader = DataLoader(dataset= val_data, batch_size = 1, shuffle=True, num_workers=16)

model = unet(3, 3, 5, 5, 7, 7, multiplier_inp, dropout_rate_inp)


# f.write("Happening till model construction")

criterioncen = CentroidLoss()
criterionseg = SoftDiceLoss()
criterionssece = Centroid_SSE_CE_Loss()
#criteriontopo = Topology_loss()

optimizercent = optim.Adam(model.parameters(), lr = 0.0000005)
optimizercen = optim.Adam(model.parameters(), lr = 0.000001)
optimizerseg = optim.Adam(model.parameters(), lr = 0.001)
optimizertopo = optim.Adam(model.parameters(), lr = 0.001)
epochs = total_epochs_inp

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# f.write("Happening before cuda loading")

# print("Device: {}".format(device))

if use_cuda:
    model = model.cuda()

# f.write("Happening after cuda loading")

seg_train_loss_array = []
box_train_loss_array = []
obox_train_loss_array = []
topo_train_loss_array = []

seg_val_loss_array = []
box_val_loss_array = []
obox_val_loss_array = []
topo_val_loss_array = []

# f.write("Happening before epoch starting")

# print("Training Started")
for epoch in range(epochs):
    # f.write("epochs started")
    start_time = time.time()
    print("Epoch: ", epoch+1)
    print("Training...")
    seg_temp_loss = []
    box_temp_loss = []
    obox_temp_loss = []
    topo_temp_loss = []

    val_seg_temp_loss = []
    val_box_temp_loss = []
    val_obox_temp_loss = []
    val_topo_temp_loss = []
    
    # if epoch == change_epoch_inp:
    #     optimizercent = optim.Adam(model.parameters(), lr = 0.00001)
    #     # optimizercen = optim.Adam(model.parameters(), lr = 0.00001)
    #     # optimizerseg = optim.Adam(model.parameters(), lr = 0.0005)
    
    for mini_batch_num, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        torch.cuda.empty_cache()
        image, mask, box, ebox, vbox, cbox = data["image"], data['mask'], data['inter_box'], data['ebox'], data['vbox'], data['cbox']
        image, mask, box, ebox, vbox, cbox  = image.to(device), mask.to(device), box.to(device), ebox.to(device), vbox.to(device), cbox.to(device)
        model.train()
        print(image.size())
        print(box.size())
        
        pred_seg, pred_box, pred_ebox, pred_vbox, pred_cbox = model(image)
        print(pred_seg.size())
        print(pred_box.size())
        
        #pred_seg_topo0 = pred_seg[:,0,:,:]
        #pred_seg_topo0 = pred_seg_topo0.view(1, -1)
        #print(pred_seg_topo0.shape)
        #mask_topo0 = mask[:,0,:,:]
        #mask_topo0 = mask_topo0.view(1, -1)
        #print(mask_topo0.shape)
        #loss_topo0 = criteriontopo(pred_seg_topo0, mask_topo0)
        
        # pred_seg_topo1 = pred_seg[:,1,:,:]
        # pred_seg_topo1 = pred_seg_topo1.view(1, -1)
        # #print(pred_seg_topo1.shape)
        # mask_topo1 = mask[:,1,:,:]
        # mask_topo1 = mask_topo1.view(1, -1)
        # #print(mask_topo1.shape)
        # loss_topo1 = criteriontopo(pred_seg_topo1, mask_topo1)

        # #pred_seg_topo2 = pred_seg[:,0,:,:]
        # #pred_seg_topo2 = pred_seg_topo2.view(1, -1)
        # #print(pred_seg_topo2.shape)
        # #mask_topo2 = mask[:,0,:,:]
        # #mask_topo2 = mask_topo2.view(1, -1)
        # #print(mask_topo2.shape)
        # #loss_topo2 = criteriontopo(pred_seg_topo2, mask_topo2)

        # loss_topo = loss_topo1

        # topo_temp_loss.append(loss_topo.item())
        # optimizertopo.zero_grad()
        # loss_topo.backward(retain_graph=True)
        # optimizertopo.step()
  
        loss_seg = criterionseg(pred_seg, mask)
        seg_temp_loss.append(loss_seg.item())

        optimizerseg.zero_grad()
        loss_seg.backward(retain_graph=True)
        optimizerseg.step()
        
        print("PEbox: ", pred_ebox.size())
        print("PVbox: ", pred_vbox.size())
        print("PCbox: ", pred_vbox.size())
        loss_obox = criterionssece(pred_ebox, pred_vbox, pred_cbox, ebox, vbox, cbox)
        obox_temp_loss.append(loss_obox.item())
        
        optimizercen.zero_grad()
        loss_obox.backward(retain_graph=True)
        optimizercen.step()

        loss_box = criterioncen(pred_box, box, factor = 10.)
        box_temp_loss.append(loss_box.item())
        
        optimizercent.zero_grad()
        loss_box.backward()
        optimizercent.step()
        
        # if (mini_batch_num + 1) % 2 == 0:
        #     print("Epoch {}/{}, MiniBatch {}/{},\tSegmnetation Loss {},\tBox Loss {}, \tOther Box Loss {}".format(epoch + 1, epochs, mini_batch_num + 1, len(train_loader), round(loss_seg.item(), 3), round(loss_box.item(), 3), round(loss_obox.item(), 3)), end = "\r", flush = True)
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            image, mask, box, ebox, vbox, cbox = data["image"], data['mask'], data['inter_box'], data['ebox'], data['vbox'], data['cbox']
            image, mask, box, ebox, vbox, cbox  = image.to(device), mask.to(device), box.to(device), ebox.to(device), vbox.to(device), cbox.to(device) 

            model.eval()

            pred_seg, pred_box, pred_ebox, pred_vbox, pred_cbox = model(image)

            loss_seg = criterionseg(pred_seg, mask)
            val_seg_temp_loss.append(loss_seg.item())
            
            # pred_val_seg_topo = pred_seg[:,1,:,:]
            # pred_val_seg_topo = pred_val_seg_topo.view(1, -1)
            # #print(pred_seg_topo1.shape)
            # mask_val_topo = mask[:,1,:,:]
            # mask_val_topo = mask_val_topo.view(1, -1)
            # loss_topo = criteriontopo(pred_val_seg_topo, mask_val_topo)
            # val_topo_temp_loss.append(loss_topo.item())

            loss_obox = criterionssece(pred_ebox, pred_vbox, pred_cbox, ebox, vbox, cbox)
            val_obox_temp_loss.append(loss_obox.item())
            
            loss_box = criterioncen(pred_box, box, factor = 10.)
            val_box_temp_loss.append(loss_box.item())

    end_time = time.time()
    
    seg_temp_loss = np.mean(np.array(seg_temp_loss))
    box_temp_loss = np.mean(np.array(box_temp_loss))
    obox_temp_loss = np.mean(np.array(obox_temp_loss))
    # topo_temp_loss = np.mean(np.array(topo_temp_loss))
    
    val_seg_temp_loss = np.mean(np.array(val_seg_temp_loss))
    val_box_temp_loss = np.mean(np.array(val_box_temp_loss))
    val_obox_temp_loss = np.mean(np.array(val_obox_temp_loss))
    # val_topo_temp_loss = np.mean(np.array(val_topo_temp_loss))
    
    seg_train_loss_array.append(seg_temp_loss)
    box_train_loss_array.append(box_temp_loss)
    obox_train_loss_array.append(obox_temp_loss)
    # topo_train_loss_array.append(topo_temp_loss)

    seg_val_loss_array.append(val_seg_temp_loss)
    box_val_loss_array.append(val_box_temp_loss)
    obox_val_loss_array.append(val_obox_temp_loss)
    # topo_val_loss_array.append(val_topo_temp_loss)

    epoch_time = end_time - start_time
    f.write("Epoch {}/{},   Time {} seconds;   Train Seg Loss {}, Train Topo Loss {},   Train Box Loss {},   Train Other Box Loss {},   Val Seg Loss {}, Val Topo Loss {},   Val Box Loss {},    Val Other Box Loss {}\n".format(epoch + 1, epochs, round(epoch_time), round(seg_temp_loss, 3), round(topo_temp_loss, 3), round(box_temp_loss, 3), round(obox_temp_loss, 3), round(val_seg_temp_loss, 3), round(val_topo_temp_loss, 3), round(val_box_temp_loss, 3), round(val_obox_temp_loss, 3)))
    print("Epoch {}/{},   Time {} seconds;   Train Seg Loss {}, Train Topo Loss {},   Train Box Loss {},   Train Other Box Loss {},   Val Seg Loss {},   Val Topo Loss {}, Val Box Loss {},    Val Other Box Loss {}\n".format(epoch + 1, epochs, round(epoch_time), round(seg_temp_loss, 3), round(topo_temp_loss, 3), round(box_temp_loss, 3), round(obox_temp_loss, 3), round(val_seg_temp_loss, 3), round(val_topo_temp_loss, 3), round(val_box_temp_loss, 3), round(val_obox_temp_loss, 3)))
    torch.save(model.state_dict(), folder + "/model_weights/"+str(epoch+1)+".pt")

plt.figure(figsize = (10, 6))
plt.plot(seg_train_loss_array, c="red", label = "Train_loss")
plt.plot(seg_val_loss_array, c="green", label = "Validation_loss")
plt.legend()
plt.savefig(folder + "/segmentation_loss")

plt.figure(figsize = (10, 6))
TL = plt.plot(box_train_loss_array, c="red", label = "Train_loss")
VL = plt.plot(box_val_loss_array, c="green", label = "Validation_loss")
plt.legend()
plt.savefig(folder + "/ir_localization_Loss")

plt.figure(figsize = (10, 6))
plt.plot(obox_train_loss_array, c="red", label = "Train_loss")
plt.plot(obox_val_loss_array, c="green", label = "Validation_loss")
plt.legend()
plt.savefig(folder + "/other_localization_loss")

# plt.figure(figsize = (10, 6))
# plt.plot(topo_train_loss_array, c="red", label = "Train_loss")
# plt.plot(topo_val_loss_array, c="green", label = "Validation_loss")
# plt.legend()
# plt.savefig(folder + "/topo_Loss")

torch.save(model.state_dict(), folder + "/checkpoints")

#TESTING PART
_, train_names = train_test_split(train_names, test_size = 0.11)

train_data = Full_Dataset("Data/images", "Data/anns", train_names, transform = transforms.Compose([ToTensor()]))

train_loader = DataLoader(dataset= train_data, batch_size = 1, shuffle=True, num_workers=16)

DOWNSIZE = 4
SCALING_FACTOR = 64

for counter in range(1, 10):
    FOLDER = folder + "/results" + str(counter)

    if "results" + str(counter) not in os.listdir(folder):
        os.mkdir(FOLDER)
        os.mkdir(FOLDER + "/train_images")
        os.mkdir(FOLDER + "/train_groundtruths")
        os.mkdir(FOLDER + "/train_detections")
        os.mkdir(FOLDER + "/train_map")
        os.mkdir(FOLDER + "/test_images")
        os.mkdir(FOLDER + "/groundtruths")
        os.mkdir(FOLDER + "/detections")
        os.mkdir(FOLDER + "/map")

    model.eval()

    with torch.no_grad():

        for i, data in enumerate(train_loader):
            if i % 1 == 0:
                ftrue = open(FOLDER + "/train_groundtruths/" + str(i) + ".txt", "w")
                fpred = open(FOLDER + "/train_detections/" + str(i) + ".txt", "w")
                image, mask, box, ebox, vbox, cbox = data["image"], data['mask'], data['inter_box'], data['ebox'], data['vbox'], data['cbox']
                image, mask, box, ebox, vbox, cbox  = image.to(device), mask.to(device), box.to(device), ebox.to(device), vbox.to(device), cbox.to(device) 


                pred_seg, pred_box, pred_ebox, pred_vbox, pred_cbox = model(image)
                
                t_pred_array, t_true_array = convert_to_numpy_pred(pred_box), convert_to_numpy(box)
    #             plot_seg_preds(t_pred_array, t_true_array, 0.001, 0)

                prd = t_pred_array[0]
                tru = t_true_array[0]

    #             im_pred = box_image(prd, DOWNSIZE, SCALING_FACTOR, 0.001)
                im_true = box_image(tru, "true", ftrue, DOWNSIZE, SCALING_FACTOR, 0.5)
                
                npred = run_non_max_suppression(prd, DOWNSIZE, SCALING_FACTOR, float(counter / 10), 0.5, 0.5)
                im_npred = box_image(npred, "pred", fpred, DOWNSIZE, SCALING_FACTOR, float(counter / 10))
                
                fig, ax = plt.subplots(1, 3, figsize = (15, 5))
                
                ax[0].imshow((im_npred + 255 * convert_torch_to_numpy(pred_seg)[0]).astype(np.uint8))
                ax[1].imshow((im_true + 255 * convert_torch_to_numpy(mask)[0]).astype(np.uint8))
                ax[2].imshow(np.transpose((image.cpu().detach().numpy()), (0, 2, 3, 1))[0].astype(np.uint8))
                plt.savefig(FOLDER + "/train_images/image_" + str(i))
                plt.close()

        for i, data in enumerate(val_loader):
            if i % 1 == 0:
                # print(i)
                ftrue = open(FOLDER + "/groundtruths/" + str(i) + ".txt", "w")
                fpred = open(FOLDER + "/detections/" + str(i) + ".txt", "w")
                image, mask, box, ebox, vbox, cbox = data["image"], data['mask'], data['inter_box'], data['ebox'], data['vbox'], data['cbox']
                image, mask, box, ebox, vbox, cbox  = image.to(device), mask.to(device), box.to(device), ebox.to(device), vbox.to(device), cbox.to(device) 

                pred_seg, pred_box, pred_ebox, pred_vbox, pred_cbox = model(image)
                
                t_pred_array, t_true_array = convert_to_numpy_pred(pred_box), convert_to_numpy(box)
    #             plot_seg_preds(t_pred_array, t_true_array, 0.001, 0)

                prd = t_pred_array[0]
                tru = t_true_array[0]

    #             im_pred = box_image(prd, DOWNSIZE, SCALING_FACTOR, 0.001)
                im_true = box_image(tru, "true", ftrue, DOWNSIZE, SCALING_FACTOR, 0.5)
                
                npred = run_non_max_suppression(prd, DOWNSIZE, SCALING_FACTOR, float(counter / 10), 0.5, 0.5)
                im_npred = box_image(npred, "pred", fpred, DOWNSIZE, SCALING_FACTOR, float(counter / 10))
                
                fig, ax = plt.subplots(1, 3, figsize = (15, 5))
                
                ax[0].imshow((im_npred + 255 * convert_torch_to_numpy(pred_seg)[0]).astype(np.uint8))
                ax[1].imshow((im_true + 255 * convert_torch_to_numpy(mask)[0]).astype(np.uint8))
                ax[2].imshow(np.transpose((image.cpu().detach().numpy()), (0, 2, 3, 1))[0].astype(np.uint8))
                plt.savefig(FOLDER + "/test_images/image_" + str(i))
                plt.close()

    print("Counter {} done".format(counter))

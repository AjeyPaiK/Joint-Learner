import numpy as np
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

def plot_seg_preds(array, true_array, threshold = 0.5, num = 0):
    fig, ax = plt.subplots(1, 2, figsize = (30, 10 * array.shape[0]))
    ax[0].imshow(array[num, :, :, 0])
    ax[1].imshow(true_array[num, :, :, 0])
    plt.show()

def box_image(a, typ, f, downsize, scaling_factor, threshold = 0.5):
    image = np.zeros((a.shape[0] * int(scaling_factor/downsize), a.shape[1] * int(scaling_factor/downsize), 3))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j][0] > threshold:
                cx, cy = int(((j) * scaling_factor + a[i][j][1] * scaling_factor) / downsize), int(((i) * scaling_factor + a[i][j][2] * scaling_factor) / downsize)    
                w, h = a[i][j][4] * scaling_factor/downsize, a[i][j][3] * scaling_factor/downsize
                rect = ((int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2)))
                if typ == "true":
                    f.write("box " + str(rect[0][0]) + " " + str(rect[0][1]) + " " + str(rect[1][0]) + " " + str(rect[1][1]) + "\n")
                elif typ == "pred":
                    f.write("box " + str(a[i][j][0]) + " " + str(rect[0][0]) + " " + str(rect[0][1]) + " " + str(rect[1][0]) + " " + str(rect[1][1]) + "\n")
                image = cv2.rectangle(image, rect[0], rect[1], (0,191,255), 5)        
    return image

def convert_to_numpy_pred(tensor):
#     print(tensor)
    tensor_cpu = tensor.cpu()
    numpy_array = tensor_cpu.detach().numpy()
    numpy_array = np.transpose(numpy_array, (0, 2, 3, 1))
    numpy_array[:, :, :, 0] = (1 / (1 + np.exp( - numpy_array[:, :, :, 0])))
    return numpy_array

def convert_to_numpy(tensor):
    tensor_cpu = tensor.cpu()
    numpy_array = tensor_cpu.detach().numpy()
    numpy_array = np.transpose(numpy_array, (0, 2, 3, 1))
    return numpy_array

def convert_torch_to_numpy(tensor):
    tensor_cpu = tensor.cpu()
    numpy_array = tensor_cpu.detach().numpy()
    numpy_array = np.transpose(numpy_array, (0, 2, 3, 1))
#     print(numpy_array.shape)
#     numpy_array = numpy_array.reshape((numpy_array.shape[0], numpy_array.shape[2], numpy_array.shape[3], numpy_array.shape[1]))
    return numpy_array


def run_non_max_suppression(centroid_array, downsize=1, scaling_factor=32, threshold=0.1, iou_threshold=0.4, encompass_threshold = 0.75):
    centr_array = compare_bounding_boxes(centroid_array, downsize, scaling_factor, threshold, iou_threshold, encompass_threshold)
    
    final = np.zeros((centroid_array.shape[0], centroid_array.shape[1], 5))
    for cinf, cimg, coord in centr_array:
        i, j = coord
        final[i, j] = cinf
    return final
    
# def compare_bounding_boxes(centroid_array, downsize, scaling_factor, threshold, iou_threshold, encompass_threshold):
#     cen_array = []
#     for i in range(centroid_array.shape[0]):
#         for j in range(centroid_array.shape[1]):
#             if centroid_array[i, j, 0] > threshold:
#                 cx, cy = int(((j) * scaling_factor + centroid_array[i,j,1])/downsize), int(((i) * scaling_factor + centroid_array[i,j,2])/downsize)    
#                 w, h = centroid_array[i,j,4] * scaling_factor/downsize, centroid_array[i,j,3] * scaling_factor/downsize
#                 rect = ((int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2)))
                
#                 img = np.zeros((centroid_array.shape[0] * scaling_factor, centroid_array.shape[1] * scaling_factor, 3))
                
#                 cv2.rectangle(img, rect[0], rect[1], (255,0,0), -1)

#                 img = img.sum(axis = -1)/255.
#                 if img.sum() > 100.:
# #                     if len(cen_array) == 0:
# #                         cen_array = [centroid_array[i, j], (i, j), img]
# #                     else:
#                     cen_array = non_max_suppression(centroid_array[i, j], (i, j), img, cen_array, iou_threshold, encompass_threshold)
#     return cen_array

def compare_bounding_boxes(centroid_array, downsize, scaling_factor, threshold, iou_threshold, encompass_threshold):
    cen_array = []
    for i in range(centroid_array.shape[0]):
        for j in range(centroid_array.shape[1]):
            if centroid_array[i, j, 0] > threshold:
                cx, cy = int(((j) * scaling_factor + centroid_array[i,j,1])/downsize), int(((i) * scaling_factor + centroid_array[i,j,2])/downsize)    
                w, h = centroid_array[i,j,4] * scaling_factor/downsize, centroid_array[i,j,3] * scaling_factor/downsize
                rect = ((int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2)))
                
                img = np.zeros((centroid_array.shape[0] * scaling_factor, centroid_array.shape[1] * scaling_factor, 3))
                
                cv2.rectangle(img, rect[0], rect[1], (255,0,0), -1)

                img = img.sum(axis = -1)/255.
                if img.sum() > 100.:
                    cen_array = non_max_suppression(centroid_array[i, j], (i, j), img, cen_array, iou_threshold, encompass_threshold)
    return cen_array

def non_max_suppression(new_inf, coordinates, img, cent_array, threshold, encompass_threshold):
    added = False
    new_added = False
    nconf = new_inf[0]
    new_cent_array = []
#     print("ninf ", new_inf)
    for cinf, cimg, coord in cent_array:
#         print("cinf ", cinf)
        conf = cinf[0]
#         print(conf, cinf, cimg.shape, coord)
        if iou(cimg, img) > threshold:            
            if conf > nconf:
                added = True
                new_cent_array.append([cinf, cimg, coord])
#             else:
#                 if not new_added:
#                     new_cent_array.append([new_inf, img, coordinates])
#                     new_added = True
        elif encompass(cimg, img, encompass_threshold):
            if conf > nconf:
                added = True
                new_cent_array.append([cinf, cimg, coord])
#             else:
#                 if not new_added:
#                     new_cent_array.append([new_inf, img, coordinates])
#                     new_added = True
        else:
            new_cent_array.append([cinf, cimg, coord])
    if not added:
        new_cent_array.append([new_inf, img, coordinates])
    del cent_array
    return new_cent_array


def iou(img1, img2):
    return ((img1 * img2).sum() + 0.01) / (img1.sum() + img2.sum() - (img1 * img2).sum() + 0.01) 

def encompass(img1, img2, encompass_threshold):
#     print(img1.sum(), img2.sum())
    minsum = min(img1.sum(), img2.sum())
    if ((img1 * img2).sum() + 0.01) / (minsum + 0.01) > encompass_threshold:
#         print(minsum, (img1 * img2).sum())
        return True
    else:
        return False
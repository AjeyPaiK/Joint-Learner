import numpy as np
import json
import cv2
import os

import skimage
from skimage.segmentation import *

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

import imutils
import matplotlib.pyplot as plt



def generate_box(polygons, downsize, scaling_factor, mirror, angle, translationx, translationy):

    box_set = np.zeros((1920//scaling_factor, 2560//scaling_factor, 5))
    
    for polygon in polygons:
        temp = np.zeros((1920, 2560), dtype=np.uint8)
        rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0])
        
        temp[rr, cc+56] = 1
        
        if mirror:
            temp = cv2.flip(temp, 1)

        if angle != 0.:
            temp = imutils.rotate(temp, angle)

        _, cnt, _ = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(cnt) > 0):
            x, y, w, h = cv2.boundingRect(cnt[0])
            cx, cy = x + w//2, y + h//2
            c = [int(cx/scaling_factor), int(cy/scaling_factor)]
            box_set[c[1], c[0]] = np.array([1., (cx % scaling_factor) / scaling_factor, (cy % scaling_factor) / scaling_factor, float(h) / scaling_factor, float(w) / scaling_factor])
    return box_set

def draw_mask(polygons, downsize):
    image = np.zeros((1920, 2560))
    
    for polygon in polygons:
        rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0])
        image[rr, cc + 56] = 1.
    image = cv2.resize(image, (image.shape[1] // downsize, image.shape[0] // downsize))
    image = np.array(image).astype(np.uint8)
    return image

def read_image_annotations(image_folder, dataset_folder, name, downsize = 4, scaling_factor = 64, mirror = False, angle = 0, translationx = 0, translationy = 0):
    height = 1920 // downsize
    width = 2560 // downsize
    
    image_filename = image_folder + "/" + name + ".jpg"
    original_image = cv2.imread(image_filename)
    
    osizes = original_image.shape

    original_image = cv2.copyMakeBorder(original_image, 0, 0, 56, 56, 0)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (width, height))
    
    original_image = np.array(original_image)
    
    name = name + ".json"
    
    ann = json.load(open(os.path.join(dataset_folder, name)))
    annotations = list(ann.values())

    coords = annotations[2]
    
    coords = [x for x in coords if type(x) == dict]

    vili_polygons = []
    dvili_polygons = []
    ccrypt_polygons = []
    hcrypt_polygons = []
    epithelium_polygons = []
    ir_polygons = []
    
    for polygon in coords:
        if polygon['label'] == 'Healthy Villi':
            vili_polygons.append(np.array(polygon['points']))
        elif polygon['label'] == 'Denudated Villi':
            dvili_polygons.append(np.array(polygon['points']))
        elif polygon['label'] == 'Epithilium':
            epithelium_polygons.append(np.array(polygon['points']))
        elif polygon['label'] == 'Healthy Crypts':
            hcrypt_polygons.append(np.array(polygon['points']))
        elif polygon['label'] == 'Circular Crypts':
            ccrypt_polygons.append(np.array(polygon['points']))
        elif polygon['label'] == 'Interpretable Region':
            ir_polygons.append(np.array(polygon['points']))
    
    
    vili = draw_mask(vili_polygons, downsize)
    dvili = draw_mask(dvili_polygons, downsize) 
    epithelium = draw_mask(epithelium_polygons, downsize)
    ccrypt = draw_mask(ccrypt_polygons, downsize)
    hcrypt = draw_mask(hcrypt_polygons, downsize)
    masks = np.concatenate([epithelium.reshape(height, width, 1), (vili + dvili).reshape(height, width, 1), (hcrypt + ccrypt).reshape(height, width, 1)], axis = -1)
    # masks = np.concatenate([epithelium.reshape(height, width, 1), vili.reshape(height, width, 1), hcrypt.reshape(height, width, 1)], axis = -1)
    if mirror:
        original_image = cv2.flip(original_image, 1)
        masks = cv2.flip(masks, 1)
        
    if angle != 0.:
        original_image = imutils.rotate(original_image, angle)
        masks = imutils.rotate(masks, angle)
    
    vili_box = generate_box(vili_polygons, downsize, scaling_factor, mirror, angle, translationx, translationy)
    dvili_box = generate_box(dvili_polygons, downsize, scaling_factor, mirror, angle, translationx, translationy)
    epithelium_box = generate_box(epithelium_polygons, downsize, scaling_factor, mirror, angle, translationx, translationy)
    ccrypt_box = generate_box(ccrypt_polygons, downsize, scaling_factor, mirror, angle, translationx, translationy)
    hcrypt_box = generate_box(hcrypt_polygons, downsize, scaling_factor, mirror, angle, translationx, translationy)
    ir_box = generate_box(ir_polygons, downsize, scaling_factor, mirror, angle, translationx, translationy)
    
    all_crypt_box = (hcrypt_box[:, :, :] + ccrypt_box[:, :, :])
    all_crypt_box[hcrypt_box[:, :, 0] == 1.] = hcrypt_box[hcrypt_box[:, :, 0] == 1.]

    all_crypt_box_pred = np.stack([hcrypt_box[:, :, 0], ccrypt_box[:, :, 0]], axis = -1)
    all_crypt_box_pred[all_crypt_box_pred.sum(axis = -1) == 2.] = [1., 0.]

    all_crypt_box_data = np.concatenate([all_crypt_box, all_crypt_box_pred], axis = -1)

    
    all_vili_box = (vili_box[:, :, :] + dvili_box[:, :, :])
    all_vili_box[vili_box[:, :, 0] == 1.] = vili_box[vili_box[:, :, 0] == 1.]

    all_vili_box_pred = np.stack([vili_box[:, :, 0], dvili_box[:, :, 0]], axis = -1)
    all_vili_box_pred[all_vili_box_pred.sum(axis = -1) == 2.] = [1., 0.]

    all_vili_box_data = np.concatenate([all_vili_box, all_vili_box_pred], axis = -1)
            
    return original_image, masks, ir_box, epithelium_box, all_crypt_box_data, all_vili_box_data, image_filename


class Full_Dataset(Dataset):
    
    def __init__(self, image_folder, dataset_folder, names, downsample = 4, scaling_factor = 64, augment = 0, flip = 0, angles = 0, image_transform = None, transform = None):
        self.image_folder = image_folder
        self.dataset_folder = dataset_folder
        
        self.names = names
        
        self.augment = augment
        self.image_transform = image_transform
        self.transform = transform
        
        self.downsample = downsample
        self.scaling_factor = scaling_factor
        
        self.augment_flip = flip
        self.augment_angles = angles
        
        self.image_transform_perm = image_transform
        self.augment_perm = augment
        
        
    def loading_times(self):
        if len(self.load_times) > 0:
            return np.mean(self.load_times)
        else:
            return 0
        
    def transforming_times(self):
        if len(self.transform_times) > 0:
            return np.mean(self.transform_times)
        else:
            return 0

    def no_noise(self, no_transform = True):
        if no_transform:
            self.image_transform = None
        else:
            self.image_transform = self.image_transform_perm
            
    def no_augment(self, no_augment = True):
        if no_augment:
            self.augment = None
        else:
            self.augment = self.augment_perm

        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, i):
        # print(self.augment)
        # print(np.random.random())

        if np.random.random() < self.augment:
            flip = False
            angle = 0
            
            if np.random.random() < self.augment_flip:
                if np.random.random() < 0.5:
                    flip = True
                else:
                    flip = False


            if np.random.random() < self.augment_angles:
                r = np.random.random()
                if r < 0.40:
                    angle = 0
                elif r < 0.80:
                    angle = 180
                elif r < 0.90:
                    angle = 90
                else:
                    angle = 270
                    
                angle += np.random.choice(range(-20, 20))

            image, mask, inter_box, ebox, cbox, vbox, fname = read_image_annotations(self.image_folder, self.dataset_folder, self.names[i], self.downsample, self.scaling_factor, mirror = flip, angle = angle)
            
        else:
            image, mask, inter_box, ebox, cbox, vbox,fname = read_image_annotations(self.image_folder, self.dataset_folder, self.names[i], self.downsample, self.scaling_factor)
        image = Image.fromarray(image)

        if self.image_transform:
            image = self.image_transform(image)
        
        sample = {"image": image, "mask": mask, "inter_box": inter_box, "ebox": ebox, "vbox": vbox, "cbox": cbox, "fname":fname}
        
        if self.transform:
            sample = self.transform(sample)
        
        
        return sample
    
class ToTensor(object):
    
    def __call__(self, sample):        
        image, mask, inter_box, ebox, vbox, cbox, fname = sample['image'], sample['mask'], sample['inter_box'], sample['ebox'], sample['vbox'], sample['cbox'], sample['fname']
        image = np.array(image)

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        inter_box = inter_box.transpose(2, 0, 1)
        
        ebox = ebox.transpose(2, 0, 1)
        vbox = vbox.transpose(2, 0, 1)
        cbox = cbox.transpose(2, 0, 1)

        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        
        mask = torch.from_numpy(mask)
        mask = mask.type(torch.FloatTensor)
        
        inter_box = torch.from_numpy(inter_box)
        inter_box = inter_box.type(torch.FloatTensor)
        
        ebox = torch.from_numpy(ebox)
        ebox = ebox.type(torch.FloatTensor)
        
        vbox = torch.from_numpy(vbox)
        vbox = vbox.type(torch.FloatTensor)
        
        cbox = torch.from_numpy(cbox)
        cbox = cbox.type(torch.FloatTensor)
        
        return {"image": image, "mask": mask, "inter_box": inter_box, "ebox": ebox, "vbox": vbox, "cbox": cbox, "fname":fname}

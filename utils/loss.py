import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, save
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data import random_split
from torch.autograd import Variable
#from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        smooth = 1.
        num = targets.size(0)
        score = 0.
        for i in range(probs.shape[1]):
            m1 = probs[:, i, :, :].view(num, -1)
            m2 = targets[:, i, :, :].view(num, -1)
            intersection = (m1 * m2)
            sc = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
            score += (1 - sc.sum() / num ) * 100. / 3.
        score = torch.mean(score)
        return score
    
    
class CentroidLoss(nn.Module):
    def __init__(self):
        super(CentroidLoss, self).__init__()

    def forward(self, preds, targets, factor = 1):
        
        loss = - 100. * (targets[:, 0, :, :] * torch.log(torch.sigmoid(preds[:, 0, :, :]) + 0.00001)) - 1. * ((1. - targets[:, 0, :, :]) * torch.log(1 - torch.sigmoid(preds[:, 0, :, :]) + 0.00001))
        
        loss += 50 * targets[:, 0, :, :] * (targets[:, 0, :, :] - torch.sigmoid(preds[:, 0, :, :])).pow(2)
        
        loss += targets[:, 0, :, :] * (targets[:, 1, :, :] - preds[:, 1, :, :]).pow(2)   

        loss += targets[:, 0, :, :] * (targets[:, 2, :, :] - preds[:, 2, :, :]).pow(2)

        loss += targets[:, 0, :, :] * (targets[:, 3, :, :] - preds[:, 3, :, :]).pow(2)

        loss += targets[:, 0, :, :] * (targets[:, 4, :, :] - preds[:, 4, :, :]).pow(2)

        loss = torch.sum(loss, dim=0)

        loss = factor * torch.mean(loss)
        
        return loss
    
    
class Centroid_SSE_CE_Loss(nn.Module):
    def __init__(self):
        super(Centroid_SSE_CE_Loss, self).__init__()

    def forward(self, pred_ebox, pred_vbox, pred_cbox, ebox, vbox, cbox):
        
        loss = self.ce_sse_loss(pred_vbox, vbox, 1.) 
        loss += self.ce_sse_loss(pred_cbox, cbox, 0.1) 
        loss += self.sse_loss(pred_ebox, ebox, 1.)
        return loss
        
        
    def ce_sse_loss(self, preds, targets, factor = 1.):
        loss = - 100. * (targets[:, 0, :, :] * torch.log(torch.sigmoid(preds[:, 0, :, :]) + 0.00001)) - 1. * ((1. - targets[:, 0, :, :]) * torch.log(1 - torch.sigmoid(preds[:, 0, :, :]) + 0.00001))
        
        loss += 50 * targets[:, 0, :, :] * (targets[:, 0, :, :] - torch.sigmoid(preds[:, 0, :, :])).pow(2)
        
        loss += targets[:, 0, :, :] * (targets[:, 1, :, :] - preds[:, 1, :, :]).pow(2)   

        loss += targets[:, 0, :, :] * (targets[:, 2, :, :] - preds[:, 2, :, :]).pow(2)

        loss += targets[:, 0, :, :] * (targets[:, 3, :, :] - preds[:, 3, :, :]).pow(2)

        loss += targets[:, 0, :, :] * (targets[:, 4, :, :] - preds[:, 4, :, :]).pow(2)
        
        loss += - 100. * targets[:, 0, :, :] * ((targets[:, 5, :, :] * torch.log(torch.sigmoid(preds[:, 5, :, :]) + 0.00001)) + ((1. - targets[:, 5, :, :]) * torch.log(1 - torch.sigmoid(preds[:, 5, :, :]) + 0.00001)))

        loss += - 100. * targets[:, 0, :, :] * ((targets[:, 6, :, :] * torch.log(torch.sigmoid(preds[:, 6, :, :]) + 0.00001)) + ((1. - targets[:, 6, :, :]) * torch.log(1 - torch.sigmoid(preds[:, 6, :, :]) + 0.00001)))

        loss += 50 * targets[:, 0, :, :] * (targets[:, 5, :, :] - torch.sigmoid(preds[:, 5, :, :])).pow(2)

        loss += 50 * targets[:, 0, :, :] * (targets[:, 6, :, :] - torch.sigmoid(preds[:, 6, :, :])).pow(2)

        loss = torch.sum(loss, dim=0)

        loss = factor * torch.mean(loss)
        
        return loss
    
    def sse_loss(self, preds, targets, factor = 1.0):
        
        loss = - 100. * (targets[:, 0, :, :] * torch.log(torch.sigmoid(preds[:, 0, :, :]) + 0.00001)) - 1. * ((1. - targets[:, 0, :, :]) * torch.log(1 - torch.sigmoid(preds[:, 0, :, :]) + 0.00001))
        
        loss += 50 * targets[:, 0, :, :] * (targets[:, 0, :, :] - torch.sigmoid(preds[:, 0, :, :])).pow(2)
        
        loss += targets[:, 0, :, :] * (targets[:, 1, :, :] - preds[:, 1, :, :]).pow(2)   

        loss += targets[:, 0, :, :] * (targets[:, 2, :, :] - preds[:, 2, :, :]).pow(2)

        loss += targets[:, 0, :, :] * (targets[:, 3, :, :] - preds[:, 3, :, :]).pow(2)

        loss += targets[:, 0, :, :] * (targets[:, 4, :, :] - preds[:, 4, :, :]).pow(2)

        loss = torch.sum(loss, dim=0)

        loss = factor * torch.mean(loss)
        
        return loss

#class Topology_loss(nn.Module):
#    def __init__(self):
#        super(Topology_loss, self).__init__()
#        self.dgminfo = LevelSetLayer2D(size=(640, 480), sublevel=False, maxdim=1)
#        self.loss = nn.MSELoss()

#    def forward(self, y_pred, y_true):
#    	intervals_in_true = self.dgminfo(y_true)
#    	l0_true = TopKBarcodeLengths(dim=0, k=20)(intervals_in_true)
#    	l1_true = TopKBarcodeLengths(dim=1, k=20)(intervals_in_true)

#    	intervals_in_pred = self.dgminfo(y_pred)
#    	l0_pred = TopKBarcodeLengths(dim=0, k=20)(intervals_in_pred)
#    	l1_pred = TopKBarcodeLengths(dim=1, k=20)(intervals_in_pred)

#    	l0 = self.loss(l0_true, l0_pred)
#    	l1 = self.loss(l1_true, l1_pred)

#    	loss = l0+l1

#    	return loss

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, save
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data import random_split
from torch.autograd import Variable

class conv_block(nn.Module):
    
    def __init__(self, ch_in, ch_out, dropout_rate):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size = 3, padding = 1),
                                 nn.Dropout2d(dropout_rate),
                                 nn.BatchNorm2d(ch_out),
                                 nn.ReLU(),
                                 nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1),
                                 nn.Dropout2d(dropout_rate),
                                 nn.BatchNorm2d(ch_out),
                                 nn.ReLU())
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class deconv_block(nn.Module):
    
    def __init__(self, ch_in, ch_out, dropout_rate):
        super(deconv_block, self).__init__()
        self.deconv = nn.Sequential((nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride = 2)), nn.Dropout2d(dropout_rate))
        
    def forward(self, x):
        x = self.deconv(x)
        return x
    
class attention_block(nn.Module):
    
    def __init__(self, ch_in, ch_skip, ch_out, dropout_rate):
        super(attention_block, self).__init__()
        self.W_skip = nn.Sequential(nn.Conv2d(ch_skip, ch_out, kernel_size = 1, padding = 0),
                                   nn.Dropout2d(dropout_rate),
                                   nn.BatchNorm2d(ch_out))
        
        self.W_in = nn.Sequential(nn.Conv2d(ch_skip, ch_out, kernel_size = 1, padding = 0),
                                   nn.Dropout2d(dropout_rate),
                                   nn.BatchNorm2d(ch_out))
        
        self.relu = nn.ReLU()
        
        self.psi = nn.Sequential(nn.Conv2d(ch_out, 1, kernel_size = 1, padding = 0),
                                nn.Dropout2d(dropout_rate),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid())        
    def forward(self, x, skip):
        # print("X:", x.size())
        # print("Skip:", skip.size())
        g = self.W_skip(skip)
        x = self.W_in(x)
        psi = self.relu(g+x)
        psi = self.psi(psi)
        return skip*psi
    

class unet(nn.Module):
    
    def __init__(self, inp_channel, out_channel, num_out1, num_out2, num_out3, num_out4, multiplier, dropout_rate):
        super(unet, self).__init__()
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=inp_channel,ch_out=8 * multiplier, dropout_rate = dropout_rate)
        self.Conv2 = conv_block(ch_in=8 * multiplier,ch_out=16 * multiplier,  dropout_rate = dropout_rate)
        self.Conv3 = conv_block(ch_in=16 * multiplier,ch_out=32 * multiplier,  dropout_rate = dropout_rate)
        self.Conv4 = conv_block(ch_in=32 * multiplier,ch_out=64 * multiplier,  dropout_rate = dropout_rate)
        self.Conv5 = conv_block(ch_in=64 * multiplier,ch_out=128 * multiplier,  dropout_rate = dropout_rate)

        self.Up5 = deconv_block(ch_in=128 * multiplier,ch_out=64 * multiplier, dropout_rate = dropout_rate)
        self.Att5 = attention_block(ch_in=64 * multiplier,ch_skip=64 * multiplier,ch_out=32 * multiplier, dropout_rate = dropout_rate)
        self.Upconv5 = conv_block(ch_in=128 * multiplier, ch_out=64 * multiplier, dropout_rate = dropout_rate)

        self.Up4 = deconv_block(ch_in=64 * multiplier,ch_out=32 * multiplier,  dropout_rate = dropout_rate)
        self.Att4 = attention_block(ch_in=32 * multiplier,ch_skip=32 * multiplier,ch_out=16 * multiplier, dropout_rate = dropout_rate)
        self.Upconv4 = conv_block(ch_in=64 * multiplier, ch_out=32 * multiplier,  dropout_rate = dropout_rate)
        
        self.Up3 = deconv_block(ch_in=32 * multiplier,ch_out=16 * multiplier,  dropout_rate = dropout_rate)
        self.Att3 = attention_block(ch_in=16 * multiplier,ch_skip=16 * multiplier,ch_out=8 * multiplier, dropout_rate = dropout_rate)
        self.Upconv3 = conv_block(ch_in=32 * multiplier, ch_out=16 * multiplier,  dropout_rate = dropout_rate)
        
        self.Up2 = deconv_block(ch_in=16 * multiplier,ch_out=8 * multiplier,  dropout_rate = dropout_rate)
        self.Att2 = attention_block(ch_in=8 * multiplier,ch_skip=8 * multiplier,ch_out=4 * multiplier, dropout_rate = dropout_rate)
        self.Upconv2 = conv_block(ch_in=16 * multiplier, ch_out=8 * multiplier,  dropout_rate = dropout_rate)

        self.Conv_1x1 = nn.Sequential(nn.Conv2d(8 * multiplier,out_channel,kernel_size=1,stride=1,padding=0), nn.Sigmoid())
        
        self.LConv1 = conv_block(ch_in=8 * multiplier,ch_out=16 * multiplier, dropout_rate = dropout_rate)
        self.LConv2 = conv_block(ch_in=16 * multiplier * 2,ch_out=32 * multiplier,  dropout_rate = dropout_rate)
        self.LConv3 = conv_block(ch_in=32 * multiplier * 2,ch_out=64 * multiplier,  dropout_rate = dropout_rate)
        self.LConv4 = conv_block(ch_in=64 * multiplier * 2,ch_out=128 * multiplier,  dropout_rate = dropout_rate)
        self.LConv5 = conv_block(ch_in=128 * multiplier * 2,ch_out=256 * multiplier,  dropout_rate = dropout_rate)

        self.LConvOut1 =  nn.Conv2d(256 * multiplier, num_out1, kernel_size=1, padding=0)
        self.LConvOut2 =  nn.Conv2d(256 * multiplier, num_out2, kernel_size=1, padding=0)
        self.LConvOut3 =  nn.Conv2d(256 * multiplier, num_out3, kernel_size=1, padding=0)
        self.LConvOut4 =  nn.Conv2d(256 * multiplier, num_out4, kernel_size=1, padding=0)
        
        self.LMaxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.LMaxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.LMaxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.LMaxpool4 = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.LMaxpool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.LAtt2 = attention_block(ch_in=16 * multiplier,ch_skip=16 * multiplier,ch_out=16 * multiplier, dropout_rate = dropout_rate)
        self.LAtt3 = attention_block(ch_in=32 * multiplier,ch_skip=32 * multiplier,ch_out=32 * multiplier, dropout_rate = dropout_rate)
        self.LAtt4 = attention_block(ch_in=64 * multiplier,ch_skip=64 * multiplier,ch_out=64 * multiplier, dropout_rate = dropout_rate)
        self.LAtt5 = attention_block(ch_in=128 * multiplier,ch_skip=128 * multiplier,ch_out=128 * multiplier, dropout_rate = dropout_rate)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool2(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool3(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool4(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        s4 = self.Att5(x = d5, skip = x4)
        d5 = torch.cat((s4,d5),dim=1)        
        d5 = self.Upconv5(d5)
        
        d4 = self.Up4(d5)
        s3 = self.Att4(x=d4,skip=x3)
        d4 = torch.cat((s3,d4),dim=1)
        d4 = self.Upconv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(x=d3,skip=x2)
        d3 = torch.cat((s2,d3),dim=1)
        d3 = self.Upconv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(x=d2,skip=x1)
        d2 = torch.cat((s1,d2),dim=1)
        d2 = self.Upconv2(d2)

        d1 = self.Conv_1x1(d2)
        
        # Localizer
        f1 = self.LConv1(d2)
        
        f2 = self.LMaxpool1(f1)
        c2 = self.LAtt2(x = f2, skip=d3)
        f2 = torch.cat((c2, f2),dim=1)
        f2 = self.LConv2(f2)
        
        f3 = self.LMaxpool2(f2)
        c3 = self.LAtt3(x = f3, skip=d4)
        f3 = torch.cat((c3, f3),dim=1)
        f3 = self.LConv3(f3)
        
        f4 = self.LMaxpool3(f3)
        c4 = self.LAtt4(x = f4, skip=d5)
        f4 = torch.cat((c4, f4),dim=1)
        f4 = self.LConv4(f4)
        
        f5 = self.LMaxpool4(f4)
        c5 = self.LAtt5(x = f5, skip=x5)
        f5 = torch.cat((c5, f5),dim=1)
        f5 = self.LConv5(f5)
        
        out1 = self.LConvOut1(f5)
        out2 = self.LConvOut2(f5)
        out3 = self.LConvOut3(f5)
        out4 = self.LConvOut4(f5)
        
        return d1, out1, out2, out3, out4
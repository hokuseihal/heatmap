import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy  as np
import os
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
def vec2img(vec,size,sp):
    #ATTENNTION!!(C,H,W)
    c=vec.shape[-1]
    vec=vec.reshape(-1,sp,sp,c)
    vec=vec.permute(0,3,2,1)
    vec=F.interpolate(vec,(size,size))
    return vec


class ImgPackModel(nn.Module):
    def __init__(self):
        super(ImgPackModel,self).__init__()
        #self.biclsmodel=PatchModel(2)
        #patchmodelsavedpath='patchmodel.pth'
        #if os.path.exists(patchmodelsavedpath):
        #    self.biclsmodel.load_state_dict(torch.load(patchmodelsavedpath))
        #    print('load',patchmodelsavedpath)

        self.feature=models.MobileNetV2(3)
        #self.cnn1=nn.Conv2d(cls,8)
        #self.cnn2=nn.Conv2d(8,4)
        #self.cnn=nn.Conv2d(4,2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1000, 2),
        )

    def forward(self,img,splittedimg,bbox,mappedbox):
        with torch.no_grad():
            splittedimg=self.biclsmodel(splittedimg)
        splittedimg=vec2img(splittedimg,128,6)
        x=torch.cat([mappedbox,splittedimg,img],dim=1)
        #x = torch.cat([mappedbox, img], dim=1)
        #x=img
        x=self.feature(x)
        x=self.classifier(x)
        return x

class PatchModel(nn.Module):
    def __init__(self, cls):
        super(PatchModel, self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=True)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, cls)

    def forward(self, x):
        # with torch.no_grad():
        s = x.shape
        x = x.reshape(-1, *s[-3:])
        x = self.cnn(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x=x.view(*s[:2],2)
        return x
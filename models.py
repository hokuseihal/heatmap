import torch
import torch.nn as nn
from mobilenet import mobilenet_v2, MobileNetV2
from resnet import resnet152
import torch.nn.functional as F
import numpy  as np
import os

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, groups=groups),
            nn.BatchNorm2d(out_planes),
            nn.Dropout2d(0.25),
            nn.ReLU()
        )


class ImgPackModel2(nn.Module):
    def __init__(self):
        super(ImgPackModel2, self).__init__()
        self.biclsmodel = PatchModel(2)
        patchmodelsavedpath = 'patchmodel.pth'
        if os.path.exists(patchmodelsavedpath):
            self.biclsmodel.load_state_dict(torch.load(patchmodelsavedpath))
            print('load', patchmodelsavedpath)

        self.feature = MobileNetV2(3)
        # self.feature=resnet152(in_channel=3)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.decider = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, img, splittedimg, bbox, mappedbox):
        x = self.feature(img)
        x = self.classifier(x)
        x = torch.cat([splittedimg, x], dim=1)
        obj = self.decider(x)
        obj = torch.softmax(obj, dim=-1)
        return obj


class ImgPackModel(nn.Module):
    def __init__(self):
        super(ImgPackModel, self).__init__()
        self.biclsmodel = PatchModel(2)
        patchmodelsavedpath = 'patchmodel.pth'
        if os.path.exists(patchmodelsavedpath):
            self.biclsmodel.load_state_dict(torch.load(patchmodelsavedpath))
            print('load', patchmodelsavedpath)

        self.feature = MobileNetV2(11)
        # self.feature=resnet152(in_channel=3)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, img, crackprobmap, bbox, mappedbox):
        x = torch.cat([img, crackprobmap, mappedbox],dim=1)
        x = self.feature(x)
        x = self.classifier(x)
        return x


class PatchModel(nn.Module):
    def __init__(self, cls):
        super(PatchModel, self).__init__()
        self.cnn = mobilenet_v2(pretrained=False)
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, cls)

    def forward(self, x):
        s = x.shape
        x = x.reshape(-1, *s[-3:])
        x = self.cnn(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = x.view(*s[:2], 2)
        return x


class SimpleImgPackModel(nn.Module):
    def __init__(self,in_ch):
        super(SimpleImgPackModel,self).__init__()
        #n,c,k
        config=[(7,8,9),(6,16,7),(6,32,5),(6,64,3),(6,128,1)]
        config=[(1,2,7),(4,4,5),(2,8,3)]
        self.features=[]
        for n,c,k in config:
            self.features.append(ConvBNReLU(in_ch,c,k))
            for _ in range(n-1):
                self.features.append(ConvBNReLU(c,c,k))
            in_ch=c
        self.features=nn.Sequential(*self.features)

        self.classifier=nn.Sequential(
            nn.Linear(8*2*2,16),
            nn.ReLU(),
            nn.Linear(16,10)
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.shape[0],-1)
        x=self.classifier(x)
        return x

import torch
import torch.nn as nn
from torchvision import models
import os
class ImgPackModel(nn.Module):
    def __init__(self):
        super(ImgPackModel,self).__init__()
        self.biclsmodel=PatchModel(2)
        patchmodelsavedpath='patchmodel.pth'
        if os.path.exists(patchmodelsavedpath):
            self.biclsmodel.load_state_dict(torch.load(patchmodelsavedpath))
            print(patchmodelsavedpath)

    def forward(self,img,splittedimg,bbox,mappedbox):
        pass

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
        return x
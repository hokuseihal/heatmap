import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from core import test
from core import train
from dataset import PatchDataset as PD
from dataset import RoadDamagePatchDataset as RDPD

# model
# trial CNN


class TrialModel(nn.Module):
    def __init__(self):
        super(TrialModel, self).__init__()
        self.cnn1 = nn.Conv2d(3, 32, 44)
        self.cnn2 = nn.Conv2d(32, 64, 44)
        self.cnn3 = nn.Conv2d(64, 128, 42)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        s = x.shape
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.cnn3(x)
        x = F.relu(x)
        x = x.view(s[0],-1)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x


class PatchModel(nn.Module):
    def __init__(self):
        super(PatchModel, self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        with torch.no_grad():
            x = self.cnn(x)
        x = self.fc(x)
        return x


# 1. pretrained CNN -> FC
# 2. CNN -> FC
# 3. pretrained CNN -> CNN
# 4. CNN -> CNN


class ClassModel(nn.Module):
    def __init__(self):
        super(ClassModel, self).__init__()
        self.patchmodel = patchmodel.cnn
        self.fc = nn.Linear(1000, 8 * 2)

    def forward(self, x):
        with torch.no_grad():
            s = x.shape()
            x = x.view(-1.0 * s[-3:])
            x = self.patchmodel(x)
        x = self.fc(x)
        x = x.view(*s[:-3])
        return x


# load data
batchsize = 32
num_epoch = 16
rdd = RDPD(rddbase="All/", patchbase="rdd_patch/", split=(6, 6))
pd = PD("rdd_patch/")
train_rdd, test_rdd = torch.utils.data.random_split(
    rdd, [int(len(rdd) * 0.7), len(rdd) - int(len(rdd) * 0.7)]
)
train_pd, test_pd = torch.utils.data.random_split(
    pd, [int(len(pd) * 0.7), len(pd) - int(len(pd) * 0.7)]
)

train_rdd_loader = torch.utils.data.DataLoader(train_rdd, batch_size=1, shuffle=True)
test_rdd_loader = torch.utils.data.DataLoader(
    test_rdd, batch_size=batchsize, shuffle=True
)
train_pd_loader = torch.utils.data.DataLoader(
    train_pd, batch_size=batchsize, shuffle=True
)
test_pd_loader = torch.utils.data.DataLoader(
    test_pd, batch_size=batchsize, shuffle=True
)
non_crack_pd = PD("rdd_patch/", non_crack=True)
test_non_crack_pd_loader = torch.utils.data.DataLoader(
    non_crack_pd, batch_size=batchsize, shuffle=True
)
# finetuning
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
patchmodel = TrialModel().to(device)
optimizer = torch.optim.Adam(patchmodel.parameters())
patchlossf = F.cross_entropy


def patchaccf(target, pred):
    return pred.eq(target.view_as(pred.long())).sum().item()


rdclossf = F.binary_cross_entropy_with_logits
# rdclossf=#TODO how ?

for e in range(1):
    train(patchmodel, device, train_pd_loader, patchlossf, optimizer, e)
    test(patchmodel, device, test_pd_loader, patchlossf, patchaccf)
    test(patchmodel, device, test_non_crack_pd_loader, patchlossf, patchaccf)
# RDD train
for e in range(num_epoch):
    train(patchmodel, device, train_rdd_loader, rdclossf, optimizer, e)
    test(patchmodel, device, test_rdd_loader, rdclossf, rdcaccf)

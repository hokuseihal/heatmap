import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from core import train, test, SoftmaxFocalLoss, SoftmaxAutoweightedTotalLoss, SoftmaxAutoweightedLoss
from dataset import RoadDamagePatchDataset as RDPD
from dataset import PatchDataset as PD


class TrialModel(nn.Module):
    def __init__(self):
        super(TrialModel, self).__init__()
        self.cnn1 = nn.Conv2d(3, 8, 44)
        self.cnn2 = nn.Conv2d(8, 16, 44)
        self.cnn3 = nn.Conv2d(16, 32, 42)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        s = x.shape
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.cnn3(x)
        x = F.relu(x)
        x = x.view(s[0], -1)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x


class PatchModel(nn.Module):
    def __init__(self, cls):
        super(PatchModel, self).__init__()
        self.cnn = models.mobilenet_v2(pretrained=True)
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
        return x


class ClassModel(nn.Module):
    def __init__(self):
        super(ClassModel, self).__init__()
        self.patchmodel = PatchModel()
        self.fc = nn.Linear(1000, 8 * 2)
        self.cnn1 = nn.Conv2d(2, 6, 3, 1, 1)

    def forward(self, x):
        s = x.shape
        # (N,Ws,Hs,C,H,W)
        x = x.view(-1, *s[-3:])
        # (N*Ws*Hs,C,H,W)
        x = self.patchmodel(x)
        # (N*Ws*Hs,2)
        x = x.view(*s[:3], 2)
        # (N,Ws,Hs,2)
        x = x.permute(0, 3, 2, 1)
        # (N,2,Hs,Ws)
        x = self.cnn1(x)
        # (N,Ws,Hs,Cls)
        return (x + 1) / 2

        # (N,Ws,Hs,Cls,conf)


# load data
batchsize = 72
num_epoch = 32
cls = 5
# rdpd = RDPD(rddbase="All/", patchbase="rdd_patch/", split=(6, 6))
rdpd = PD('rdd_patch/')
train_rdd, test_rdd = torch.utils.data.random_split(
    rdpd, [int(len(rdpd) * 0.7), len(rdpd) - int(len(rdpd) * 0.7)]
)

train_rdpd_loader = torch.utils.data.DataLoader(
    train_rdd, batch_size=batchsize, shuffle=True
)
test_rdpd_loader = torch.utils.data.DataLoader(
    test_rdd, batch_size=batchsize, shuffle=True
)
# finetuning
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
patchmodel = PatchModel(cls).to(device)
optimizer = torch.optim.Adam(patchmodel.parameters(), lr=1e-5)
patchlossf = nn.CrossEntropyLoss()
patchlossf = nn.CrossEntropyLoss(weight=torch.Tensor([0.01, 0.5, 0.33, 0.33, 1]).to(device))
patchlossf = SoftmaxAutoweightedLoss(cls)
patchlossf = SoftmaxAutoweightedTotalLoss(cls)
patchlossf = SoftmaxFocalLoss()


def patchaccf(target, pred):
    return pred.eq(target.view_as(pred.long())).sum().item()


def rdcaccf(target, pred, limit=0.5):
    tp = target.bool() & (pred > limit)
    fp = (~target.bool()) & (pred > limit)
    tn = target.bool() & ~(pred > limit)
    fn = ~target.bool() & ~(pred > limit)

    return torch.stack(
        [
            tp.view(-1, tp.shape[-1]).sum(0),
            fp.view(-1, fp.shape[-1]).sum(0),
            tn.view(-1, tn.shape[-1]).sum(0),
            fn.view(-1, fn.shape[-1]).sum(0),
        ]
    )


def prmap(target, pred):
    num_cls = pred.shape[-1]
    numbatch = pred.shape[0]
    rmap = torch.zeros((num_cls, num_cls))
    for i in range(numbatch):
        rmap[target[i], pred[i].argmax()] += 1
    return rmap


test(patchmodel, device, test_rdpd_loader, patchlossf, patchaccf, prmap)
for e in range(num_epoch):
    train(patchmodel, device, train_rdpd_loader, patchlossf, optimizer, e)
    test(patchmodel, device, test_rdpd_loader, patchlossf, patchaccf, prmap)
torch.save(patchmodel, 'patchmodel.pth')

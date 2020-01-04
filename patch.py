import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from core import yolotrain, yolotest, train, test, SoftmaxFocalLoss, SoftmaxAutoweightedTotalLoss, \
    SoftmaxAutoweightedLoss, FocalLoss
from dataset import YOLOcatPatchDataset as YPD
from dataset import PatchDataset as PD

import os


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


class YoloPatchmodel(nn.Module):
    def __init__(self, cls):
        super(YoloPatchmodel, self).__init__()
        self.patchmodel = PatchModel(cls)
        if os.path.exists('patchmodel.pth'):
            self.patchmodel.load_state_dict(torch.load('patchmodel.pth'))
        self.cnn1 = nn.Conv2d(35, 36, 4)
        self.cnn2 = nn.Conv2d(36, 64, 3)
        self.cnn3 = nn.Conv2d(64, 128, 3)
        self.cnn4 = nn.ConvTranspose2d(130, 64, 3)
        self.cnn5 = nn.ConvTranspose2d(64, 36, 3)
        self.cnn6 = nn.ConvTranspose2d(36, 35, 4)

    def forward(self, img, yolo_out_cls_conf):
        img_shape = img.shape
        yolo_shape = yolo_out_cls_conf.shape
        img = img.view(-1, *img_shape[-3:])
        x = self.patchmodel(img)
        x = x.view(-1, 6, 6, 2)
        x = x.permute(0, 3, 2, 1)
        # x:(B,sp,sp,C,H,W)
        yolo_out_cls_conf = yolo_out_cls_conf.view(-1, 13, 13, 35)
        yolo_out_cls_conf = yolo_out_cls_conf.permute(0, 3, 2, 1)
        yolo_out_cls_conf = self.cnn1(yolo_out_cls_conf)
        yolo_out_cls_conf = F.relu(yolo_out_cls_conf)
        yolo_out_cls_conf = self.cnn2(yolo_out_cls_conf)
        yolo_out_cls_conf = F.relu(yolo_out_cls_conf)
        yolo_out_cls_conf = self.cnn3(yolo_out_cls_conf)
        # x:(B,6,6,2)
        x = torch.cat([x, yolo_out_cls_conf], dim=1)
        x = self.cnn4(x)
        x = F.relu(x)
        x = self.cnn5(x)
        x = F.relu(x)
        x = self.cnn6(x)
        x = x.permute(0, 3, 2, 1)
        x = x.view(-1, 13, 13, 5, 7)
        #x = torch.sigmoid(x)
        return x


# load data
batchsize = 72
num_epoch = 252
cls = 2
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
# TODO load train.txt
train_ypd = YPD('All/', 'pickle.pkl','All/ImageSets/Main/train.txt')
test_ypd=YPD('All/', 'pickle.pkl','All/ImageSets/Main/val.txt')
train_ypd_loader = torch.utils.data.DataLoader(
    train_ypd, batch_size=4, shuffle=False
)
test_ypd_loader = torch.utils.data.DataLoader(
    test_ypd, batch_size=4, shuffle=False
)
# finetuning
device = "cuda" if torch.cuda.is_available() else "cpu"
# device='cpu'
print(device)
patchmodel = PatchModel(cls).to(device)
yolopatchmodel = YoloPatchmodel(cls).to(device)
optimizer = torch.optim.Adam(patchmodel.parameters())
yolooptimizer=torch.optim.Adam(yolopatchmodel.parameters())
# patchlossf = nn.CrossEntropyLoss(weight=torch.Tensor([0.01, 0.5, 0.33, 0.33, 1]).to(device))
# patchlossf = SoftmaxAutoweightedLoss(cls)
# patchlossf = SoftmaxAutoweightedTotalLoss(cls)
# patchlossf = nn.CrossEntropyLoss()
patchlossf = SoftmaxFocalLoss(gammma=2)
yolocatpatchlossf = FocalLoss()


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


from core import readanchors
from torchvision.ops import nms
import csv


def nmswritecsv(xy, wh, clsconf, imgname, thresh=0.5):
    boxes = torch.cat([xy * 600, wh * torch.from_numpy(readanchors())], dim=-1)
    # convert (x,y,w,h) -> (x0,y0,x1,y1)
    x0 = boxes[:, :, :, :, :, 0] - boxes[:, :, :, :, :, 2] / 2
    y0 = boxes[:, :, :, :, :, 1] - boxes[:, :, :, :, :, 3] / 2
    x1 = boxes[:, :, :, :, :, 0] + boxes[:, :, :, :, :, 2] / 2
    y1 = boxes[:, :, :, :, :, 1] + boxes[:, :, :, :, :, 3] / 2
    x0 = x0.view(*x0.shape, 1)
    y0 = y0.view(*y0.shape, 1)
    x1 = x1.view(*x1.shape, 1)
    y1 = y1.view(*y1.shape, 1)
    boxes = torch.cat([x0, y0, x1, y1], dim=-1)
    score ,cls= clsconf[:, :, :, :,:-1].max(dim=-1)
    score*=clsconf[:,:,:,:,-1]
    boxes=boxes.view(-1, 4)
    score=score.view(-1)
    cls=cls.view(-1,1)
    ids = nms(boxes, score, 0.5)
    with open('catyolo.csv', 'w') as f:
        writer = csv.writer(f)
        for i in ids:
            if score[i] > thresh:
                writer.writerow([imgname[i//(13*13*5)].split('.')[0], cls[i].item(), score[i].item(), *boxes[i].int().tolist()])

# test(patchmodel, device, test_rdpd_loader, patchlossf, patchaccf, prmap)
##pretraining for patch binary classification
# for e in range(num_epoch):
#   train(patchmodel, device, train_rdpd_loader, patchlossf, optimizer, e)
#   test(patchmodel, device, test_rdpd_loader, patchlossf, patchaccf, prmap)
#   torch.save(patchmodel.state_dict(), 'patchmodel.pth')

for e in range(num_epoch):
    yolotrain(yolopatchmodel, device, train_ypd_loader, yolocatpatchlossf, yolooptimizer, e)
    yolotest(yolopatchmodel, device, test_ypd_loader, yolocatpatchlossf, patchaccf, nmswritecsv)
    torch.save(patchmodel.state_dict(), 'yolopatchmodel.pth')

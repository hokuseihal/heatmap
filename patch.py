import os

import torch
import torch.nn.functional as F
from torch import nn

from core import train, test, SoftmaxFocalLoss, FocalLoss
from dataset import PatchDataset as PD
from models import PatchModel
from check_2cls_cut import check_2cls_cut

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
        self.cnn6 = nn.ConvTranspose2d(36, 7, 4)

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
        x = F.relu(x)
        x = x.permute(0, 3, 2, 1)
        x = x.view(-1, 13, 13, 7)
        # x = torch.sigmoid(x)
        return x


# load data
batchsize = 128
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
    test_rdd, batch_size=1, shuffle=True
)
# train_ypd = YPD('All/', 'pickle.pkl','All/ImageSets/Main/train.txt')
# test_ypd=YPD('All/', 'pickle.pkl','All/ImageSets/Main/val.txt')
# train_ypd_loader = torch.utils.data.DataLoader(
#    train_ypd, batch_size=4, shuffle=True
# )
# test_ypd_loader = torch.utils.data.DataLoader(
#    test_ypd, batch_size=1, shuffle=True
# )
# finetuning
device = "cuda" if torch.cuda.is_available() else "cpu"
# device='cpu'
print(device)
patchmodel = PatchModel(cls).to(device)
if os.path.exists('patchmodel.pth'):
    patchmodel.load_state_dict(torch.load('patchmodel.pth'))
    print('load patch weight')
yolopatchmodel = YoloPatchmodel(cls).to(device)
# if os.path.exists('yolopatchmodel.pth'):
#    yolopatchmodel.load_state_dict(torch.load('yolopatchmodel.pth'))
#    print('load yolo weight')
optimizer = torch.optim.Adam(patchmodel.parameters())
# yolooptimizer=torch.optim.Adam(yolopatchmodel.parameters())
patchlossf = SoftmaxFocalLoss(gammma=2)
yolocatpatchlossf = FocalLoss()

from core import patchaccf


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


from core import prmap

from core import readanchors
from torchvision.ops import nms
import csv


def nmswritecsv(xy, wh, clsconf, imgname, thresh=0.1):
    print(imgname)
    boxes = torch.cat([xy * 600, wh * torch.from_numpy(readanchors()) * 600 / 13], dim=-1)
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
    clsconf = clsconf.view(clsconf.shape[0], 13, 13, 1, 7)
    clsconf = torch.ones((clsconf.shape[0], 13, 13, 5, 7)).to(device) * clsconf
    score, cls = clsconf[:, :, :, :, :-1].max(dim=-1)
    score *= clsconf[:, :, :, :, -1]
    boxes = boxes.view(-1, 4)
    score = score.view(-1)
    cls = cls.view(-1, 1)
    ids = nms(boxes.cpu(), score.cpu(), 0.5)
    with open('catyolo.csv', 'a') as f:
        writer = csv.writer(f)
        for i in ids:
            if score[i] > thresh:
                print(imgname[i // (13 * 13 * 5)].split('.')[0], cls[i].item(), score[i].item(),
                      *boxes[i].int().tolist())
                writer.writerow([imgname[i // (13 * 13 * 5)].split('.')[0], cls[i].item(), score[i].item(),
                                 *boxes[i].int().tolist()])


# pretraining for patch binary classification
mx = 0
for e in range(num_epoch):
    train(patchmodel, device, train_rdpd_loader, patchlossf, optimizer, e)
    test(patchmodel, device, test_rdpd_loader, patchlossf, patchaccf, prmap)
    r=check_2cls_cut(patchmodel,device)
    print('recall',r)
    if mx<r:
       mx=r
       print('BIGGER!!!\n',mx)
       torch.save(patchmodel.state_dict(), 'patchmodel.pth')
# savedic={}
# from loadimg import  loadimgsp
# import pickle
# txt='/home/hokusei/src/mydarknet/all.txt'
# with open(txt) as f:
#    lines=[i.strip() for i in f.readlines()]
# for l in lines:
#    img=loadimgsp(l)
#    img=img.to(device)
#    patchmodel.eval()
#    with torch.no_grad():
#        out=patchmodel(img)
#    out=out.view(6,6,2)
#    out=torch.softmax(out,dim=-1)
#    savedic[l]=out
# with open('patch.pkl','wb') as f:
#    pickle.dump(savedic,f)

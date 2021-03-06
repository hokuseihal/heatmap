import torch
import torch.nn as nn
import torch.nn.functional as F
import os
classname = ['D00', 'D01', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44', 'D30']
def readxml(filename,root='All/Annotations/'):
    filename=root+filename.strip().split('/')[-1].split('.')[0]+'.xml'
    global readclslist
    r_list = []
    try:
        tree = ET.parse(filename)

        root = tree.getroot()
        for child in root.iter('object'):
            for object in child:
                if 'name' == object.tag:
                    cls = object.text

                if 'bndbox' == object.tag:
                    r_list.append([int(xy.text) for xy in object] + [classname.index(cls)])
        return r_list

    except FileNotFoundError:
        print(f'{filename} is not Found')
        return []
def patchaccf(target, pred):
    return pred.eq(target.view_as(pred.long())).sum().item()

def prmap(target, pred):
    num_cls = pred.shape[-1]
    numbatch = pred.shape[0]
    rmap = torch.zeros((num_cls, num_cls))
    for i in range(numbatch):
        rmap[target[i], pred[i].argmax()] += 1
    return rmap
def cal_iou(r, x):
    if not (r[0] < x[2] and x[0] < r[2] and r[1] < x[3] and x[1] < r[3]):
        return 0

    A = (x[2] - x[0]) * (x[3] - x[1])
    B = (r[2] - r[0]) * (r[3] - r[1])
    C = (min(r[2], x[2]) - max(r[0], x[0])) * (min(r[3], x[3]) - max(r[1], x[1]))
    iou = C / (A + B - C)
    return iou
def ospathcat(namelist):
    s=''
    for name in namelist:
        s=os.path.join(s,name)
    return s
class SoftmaxFocalLoss(torch.nn.Module):
    def __init__(self, gammma=2, average=True):
        super(SoftmaxFocalLoss, self).__init__()
        self.gammma = gammma
        self.average = average

    def forward(self, input, target):
        '''
        input:(B,C) :float32
        target(B)   :Long
        output(1)      :float32
        '''
        target = target.reshape(-1)
        assert input.dtype == torch.float
        assert target.dtype == torch.long
        assert len(input.shape) == 2
        assert input.shape[0] == target.shape[0]
        pt = F.softmax(input, dim=-1)
        logpt = F.log_softmax(input, dim=-1)
        input = -((1 - pt) ** self.gammma) * logpt
        input = input[range(input.shape[0]), target]
        if self.average:
            return input.mean()
        else:
            return input.sum()


class SoftmaxAutoweightedLoss(torch.nn.Module):
    def __init__(self, cls):
        super(SoftmaxAutoweightedLoss, self).__init__()
        self.cls = cls

    def forward(self, input, target):
        '''
        input:(B,C) :float32
        target(B)   :Long
        output(1)      :float32
        '''
        target = target.reshape(-1)
        assert input.dtype == torch.float
        assert target.dtype == torch.long
        assert len(input.shape) == 2
        assert input.shape[0] == target.shape[0]
        c = torch.stack([(1 - target).sum(), target.sum()]).float()
        c = c.min() / c + 1e-5
        lossf = torch.nn.CrossEntropyLoss(weight=c)
        return lossf(input, target)


class SoftmaxAutoweightedTotalLoss(torch.nn.Module):
    def __init__(self, cls):
        super(SoftmaxAutoweightedTotalLoss, self).__init__()
        self.gamma = 2
        self.cls = cls

    def forward(self, input, target):
        '''
        input:(B,C) :float32
        target(B)   :Long
        output(1)      :float32
        '''
        target = target.reshape(-1)
        assert input.dtype == torch.float
        assert target.dtype == torch.long
        assert len(input.shape) == 2
        assert input.shape[0] == target.shape[0]
        c = torch.stack([(target == i).sum() for i in range(self.cls)]).float()
        c = c.min() / c + 1e-5
        pt = F.softmax(input, dim=-1)
        logpt = F.log_softmax(input, dim=-1)
        input = -((1 - pt) ** self.gamma) * logpt
        input = input[range(input.shape[0]), target] * torch.stack([c[t] for t in target])
        return input.mean()


class FocalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = 2
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, input, target):
        assert input.shape == target.shape
        #assert 0 <= input.min()
        #assert 1 >= input.max()
        #assert 0 <= target.min()
        #assert 1 >= target.max()
        def focal(_loss,gamma):
            return _loss**gamma
        loss=self.mse(input,target)
        loss = focal(loss,self.gamma)
        return loss.sum()


def train(model, device, train_loader, lossf, optimizer, epoch, log_interval=1):
    model.train()
    tloss = 0
    #log_interval = len(train_loader)
    log_interval=3
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float32), target.to(device).reshape(-1)
        optimizer.zero_grad()
        output = model(data)
        loss = lossf(output, target)
        loss.backward()
        optimizer.step()
        tloss = tloss + loss.item()

        if (batch_idx + 1) % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    tloss / log_interval,
                )
            )
            tloss = 0
        #if batch_idx==3:
        #    break
def yolotrain(model, device, train_loader, lossf, optimizer, epoch, log_interval=100):
    model.train()
    tloss =[]
    #log_interval = len(train_loader)
    for batch_idx, (img,data, target,_,_,_) in enumerate(train_loader):
        model=model.to(device)
        img,data, target = img.to(device),data.to(device, dtype=torch.float32), target.to(device)
        optimizer.zero_grad()
        output = model(img,data)
        loss = lossf(output, target)
        loss.backward()
        optimizer.step()
        #print(f'{loss.item():.6f}')
        tloss.append(loss.item())
        if (batch_idx + 1) % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx:5d}/{len(train_loader)} ({batch_idx/len(train_loader)*100:.0f}%)]\tLoss: {np.mean(tloss)}')
            tloss = []
import time
def test(model, device, test_loader, lossf, accf, prf):
    # accf:input:*labels,*labels
    #    :return:number of TP
    model.eval()
    test_loss = 0
    correct = 0
    rmap = 0
    with torch.no_grad():
        for idx,(data, target) in enumerate(test_loader):
            data, target = data.to(device, dtype=torch.float32), target.to(device).reshape(-1)
            output = model(data)
            # sum up batch loss
            test_loss += lossf(output, target)
            # get the index of the max log-probability

            pred = output.argmax(dim=-1, keepdim=True)
            correct += accf(target, pred)
            rmap += prf(target, output)
            #print(idx,'/',len(test_loader),end='')

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss / len(test_loader),
            correct,
            len(test_loader.dataset),
            100.0 * correct / (len(test_loader.dataset)),
        )
    )

    print(f'precision:{rmap.diag() / rmap.sum(dim=-1)}\nrecall:{rmap.diag() / rmap.sum(dim=0)}\n\n')
    if (rmap.diag() / rmap.sum(dim=-1))[1]>0.68 and (rmap.diag() / rmap.sum(dim=0))[1]>0.68:
        exit(0)

def yolotest(model, device, test_loader, lossf, accf, prf):
    model=model.to(device)
    model.eval()
    test_loss = []
    with torch.no_grad():
        for img,data, target,yolo_xy,yolo_wh,imgname in test_loader:
            img, data, target = img.to(device), data.to(device, dtype=torch.float32), target.to(device)
            output = model(img,data)
            test_loss .append( lossf(output, target).item())
            prf(yolo_xy,yolo_wh,output,imgname)

    print(f'Test set: Average loss: {np.mean(test_loss)}')



classes = ['D00', 'D01', 'D10', 'D11', 'D20', 'D40']
import xml.etree.ElementTree as ET
import numpy as np

def readanchors(anchors_path='yolo_anchors.txt'):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        return anchors.astype(np.float32)
def xml2clsconf(path, split):
    clsconf = np.zeros((split, split,len(classes) + 1),dtype=np.float32)
    anchors=readanchors()
    try:
        tree = ET.parse(path)
        width = None
        height = None
        cls = None
        root = tree.getroot()
        for child in root.iter('size'):
            for size in child:
                if 'width' == size.tag:
                    width = int(size.text)
                elif 'height' == size.tag:
                    height = int(size.text)
        for child in root.iter('object'):
            for object in child:
                if 'name' == object.tag:
                    cls = object.text
                    if not cls in classes: break
                if 'bndbox' == object.tag:
                    xmin, ymin, xmax, ymax = [int(xy.text) for xy in object]
                    def nearest(_point,_anchors):
                        anc=((_anchors-_point)**2).mean(axis=-1).argmin()
                        return anc
                    anc=nearest(((xmax-xmin)*13/width,(ymax-ymin)*13/height),anchors)
                    spx, spy = int((xmin + xmax) / (2 * width) * split ), \
                               int((ymin + ymax) / (2 * height) * split )
                    clsconf[spx, spy, classes.index(cls)] = True
                    clsconf[spx, spy, -1] = True

        return clsconf

    except FileNotFoundError:
        print(f'{path} is not Found')
def base(path):
    return path.split('/')[-1].split('.')[0]
def getbb(basename, xmlbase="All/Annotations/", normalize=True):
    bb = []
    with open(xmlbase + basename + ".xml") as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        t = list(root.iter("object"))
        if len(t) == 0:
            return []
        #    pass
        for obj in root.iter("object"):
            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text) / w,
                float(xmlbox.find("ymin").text) / h,
                float(xmlbox.find("xmax").text) / w,
                float(xmlbox.find("ymax").text) / h,
            )
            if normalize:
                pass

            else:

                b = (
                    float(xmlbox.find("xmin").text),
                    float(xmlbox.find("ymin").text),
                    float(xmlbox.find("xmax").text),
                    float(xmlbox.find("ymax").text),
                )
            cls = classes.index(obj.find("name").text)
            bb.append((cls, b))
        return bb
def checkRDD(path, bbox,param='TF',iouthresh=0.5):
    def calscore(box1, box2):
        cls1, box1 = box1
        cls2, box2 = box2
        if param == 'TF':
            return cal_iou(box1, box2) > iouthresh  # and cls1==cls2
    bboxes = getbb(base(path), normalize=False)
    scorelist = [calscore(bbox, b) for b in bboxes]
    if param == 'TF':
        return False if scorelist == [] else max(scorelist) == True
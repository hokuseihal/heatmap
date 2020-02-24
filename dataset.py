import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
import csv
from loadimg import Load2cls
from PIL import Image
from core import ospathcat
def vec2img(vec, size, sp):
    # ATTENNTION!!(C,H,W)
    c = vec.shape[-1]
    vec = vec.reshape(-1, sp, sp, c)
    vec = vec.permute(0, 3, 2, 1)
    vec = F.interpolate(vec, (size, size))
    return vec


def noex(path):
    return os.path.splitext(path)[0]


def base(path):
    return path.split('/')[-1].split('.')[0]


def file(path):
    return path.split('/')[-1]


class YOLOOutputDataset(torch.utils.data.Dataset):
    def __init__(self, base, csvpath, size=(128, 128), numcls=8,iouthresh=.5, crop=True, prob_thresh=0.5, param='TF',mappedcls=True,getmapimg=False):
        self.crop = crop
        self.base = base
        self.size = size
        self.iouthresh = iouthresh
        self.param = param
        with open(csvpath) as f:
            csvreader = csv.reader(f)
            self.yolooutput = [row for row in csvreader if int(row[1]) < 8 and float(row[2]) > prob_thresh]
        self.transform = Compose([Resize(size), ToTensor()])
        self.bicls = Load2cls('patch.pkl')
        self.c = []
        self.mappedcls=mappedcls
        self.getmapimg=getmapimg
        self.numcls=numcls

    def __len__(self):
        return len(self.yolooutput)

    def __getitem__(self, idx):
        imgname = ospathcat([self.base, 'JPEGImages', file(self.yolooutput[idx][0].strip())])
        if not '.jpg' in imgname:
            imgname+='.jpg'
        objecet_detection_api=True
        cls = int(self.yolooutput[idx][1]) if not objecet_detection_api else int(self.yolooutput[idx][1])-1
        prob = float(self.yolooutput[idx][2])
        x0 = int(self.yolooutput[idx][3])
        y0 = int(self.yolooutput[idx][4])
        x1 = int(self.yolooutput[idx][5])
        y1 = int(self.yolooutput[idx][6])
        img = Image.open(imgname)
        if self.crop:
            img = img.crop((x0, y0, x1, y1))
            # img.save('img.jpg','JPEG', optimize=True)

        splitedimg = self.bicls.get(imgname)
        if self.getmapimg:
            splitedimg=vec2img(splitedimg,128,6)[0]
        else:
            splitedimg = (splitedimg[x0 // 100:(x1 // 100) + 1, y0 // 100:(y1 // 100) + 1]).reshape(-1, 2).mean(
                dim=0).reshape(2)
        if (splitedimg != splitedimg).all():
            splitedimg = torch.zeros_like(splitedimg)
        img = self.transform(img)
        # Attention! (C,H,W)
        obj = self.checkRDD(imgname, (cls, (x0, y0, x1, y1)))
        x0 = int(x0 * self.size[0] / 600)
        y0 = int(y0 * self.size[1] / 600)
        x1 = int(x1 * self.size[0] / 600)
        y1 = int(y1 * self.size[1] / 600)
        if self.mappedcls:
            mapped_box = torch.zeros(1, *self.size)
            mapped_box[0, x0:x1, y0:y1] = prob
        else:
            # TODO BE VARIABLE
            mapped_box = torch.zeros(self.numcls, *self.size)
            mapped_box[cls, x0:x1, y0:y1] = prob
        # if obj:
        #    if (splitedimg==splitedimg).all():
        #        self.c.append(splitedimg[1].item())
        return img, splitedimg, mapped_box, (cls, prob, x0, y0, x1, y1), int(obj), idx

    def checkRDD(self, path, bbox):
        def calscore(box1, box2):
            cls1, box1 = box1
            cls2, box2 = box2
            if self.param == 'TF':
                return cls1==cls2 and cal_iou(box1, box2) > self.iouthresh  # and cls1==cls2
            elif self.param == 'REG':
                return (cls1 == cls2) * (cal_iou(box1, box2) > self.reg_iou_thresh) * cal_iou(box1, box2)

        bboxes = getbb(base(path), normalize=False)
        scorelist = [calscore(bbox, b) for b in bboxes]

        if self.param == 'TF':
            return False if scorelist == [] else max(scorelist) == True


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

'''
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, base, rddbase='All/', in_transform=None, split=(6, 6)):
        self.transform = (
            Compose([Resize((128, 128)), ToTensor()])
            if in_transform is None
            else in_transform
        )
        self.positive_base = base + "Positive/"
        self.negative_base = base + "Negative/"
        self.negative_road_base = base + "Negative_road/"
        self.imagelist = (
                [
                    (self.positive_base + path, pos2cls_cof(rddbase, *(split_head_num(path)), split))
                    for path in os.listdir(self.positive_base)
                ]
                + [
                    (self.negative_base + path, 0)
                    for path in (os.listdir(self.negative_base))
                ]
                + [
                    (self.negative_road_base + path, 0)
                    for path in os.listdir(self.negative_road_base)
                ]
        )
        print(f"Patch:{len(self.imagelist)}")

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        impath, label = self.imagelist[idx]
        return self.transform(Image.open(impath)), label


'''
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, base, in_transform=None, ratio=0.5, non_crack=False):
        self.transform = (
            Compose([Resize((128, 128)), ToTensor()])
            if in_transform is None
            else in_transform
        )

        self.positive_base = base + "Positive/"
        self.negative_base = base + "Negative/"
        self.negative_road_base = base + "Negative_road/"
        if non_crack:
            self.imagelist = [
                (self.negative_road_base + path, 0)
                for path in sorted(os.listdir(self.negative_road_base))[
                            int(ratio * len(os.listdir(self.negative_road_base))):
                            ]
            ]
        else:
            self.imagelist = (
                    [
                        (self.positive_base + path, 1)
                        for path in os.listdir(self.positive_base)
                    ]
                    + [
                        (self.negative_base + path, 0)
                        for path in (os.listdir(self.negative_base))[
                                    : int((1 - ratio) * len(os.listdir(self.positive_base)))
                                    ]
                    ]
                    + [
                        (self.negative_road_base + path, 0)
                        for path in sorted(os.listdir(self.negative_road_base))[
                                    : int(ratio * len(os.listdir(self.negative_road_base)))
                                    ]
                    ]
            )
        print(f"Patch:{len(self.imagelist)}")

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        impath, label = self.imagelist[idx]
        return self.transform(Image.open(impath)), label


# RDD dataset
classes = ["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44", "D30"]
num_cls = len(classes)

'''
class RDcDataset(torch.utils.data.Dataset):
    # just convert bbox to tensor
    # this contains non_crack image as crack
    # take care on using this dataset
    def __init__(self, base, transforms=None, r="tensor", split=None):
        if split is None:
            self.transforms = ToTensor() if transforms is None else transforms
        else:
            self.transforms = (
                PIL2Tail(*split, "torch") if transforms is None else transforms
            )
        self.base = base
        self.images = list(
            set([noex(xml) for xml in os.listdir(base + "Annotations")])
            & set([noex(im) for im in os.listdir(base + "JPEGImages")])
        )
        print(f"RDD:{len(self.images)}")
        self.r = r
        self.split = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        basename = self.images[idx]
        bb = getbb(basename, normalize=False)
        im = Image.open(self.base + "JPEGImages/" + basename + ".jpg")
        if self.r == "bb":
            return self.transforms(im), bb
        if self.r == "tensor":
            r_tensor = torch.zeros((*(self.split), len(classes)))
            for cls, b in bb:
                r_tensor[
                round(b[0] / w_d): round(b[1] / w_d),
                round(b[2] / h_d): round(b[3] / h_d),
                cls,
                ] = 1
            return self.transforms(im), r_tensor
'''


def split_head_num(s):
    import re

    return int(re.match("^\d+", s).group()), re.sub("^\d+", "", s).split(".")[0]


def isinbox(pos, sp, b):
    for coor in b:
        assert 0 <= coor <= 1, "BBOX must be normalized 0~1... Check argument"
    return (
        True
        if (b[0] <= (pos % sp[0] + 0.5) / sp[0] <= b[1])
           and (b[2] <= ((pos // sp[1] + 0.5) / sp[1]) <= b[3])
        else False
    )


def pos2posvec(base, imp, pos, sp):
    basename = imp.split(".")[0]
    bb = getbb(basename, xmlbase=base + "Annotations/")
    r_tensor = torch.zeros(num_cls)
    for cls, b in bb:
        r_tensor[cls] = 1.0 if isinbox(pos, sp, b) else 0.0
    return r_tensor


c = 0


def cal_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def pos2coor(pos, sp):
    x, y = num2postuple(pos, sp)
    return (x / sp[0], y / sp[1], (x + 1) / sp[0], (y + 1) / sp[1])


def pos2cls_cof(base, pos, imp, sp):
    global c
    basename = imp.split(".")[0]
    bb = getbb(basename, xmlbase=base + "Annotations/")
    preiou = 0
    rcls = 1
    for cls, b in bb:
        iou = cal_iou(b, pos2coor(pos, sp))
        if preiou < iou:
            if preiou != 0: c += 1
            preiou = iou
            rcls = cls
    assert rcls != 0

    def capsle(cls):
        l = [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]
        l = [0, 1, 2, 2, 2, 2, 3, 4, 1, 1, 1]
        l = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # print(l[cls])
        return l[cls]

    rcls = capsle(rcls)
    return rcls


def num2postuple(num, sp):
    return num % sp[0], num // sp[0]


# TODO DEBUG
from loadimg import loadimgsp


class RoadDamagePatchDataset(torch.utils.data.Dataset):
    def __init__(self, rddbase, patchbase, split):
        assert isinstance(split, tuple), "argment sp's type must be TUPLE."
        assert len(split) == 2, "argment sp's length must be 2."
        self.rddbase = rddbase
        self.patchbase = patchbase
        self.namelist = []
        self.num_cls = num_cls
        foll = ["Positive", "Negative", "Negative_road"]
        for fol in foll:
            for imp in os.listdir(patchbase + fol):
                num, basename = split_head_num(imp)
                if basename not in self.namelist:
                    self.namelist.append(basename)
        self.targetl = torch.zeros((len(self.namelist), *split))

        # on each num ,get bbox info and restore if patch is positive
        for imp in os.listdir(patchbase + "Positive"):
            num, basename = split_head_num(imp)
            self.targetl[
                (self.namelist.index(basename), *num2postuple(num, split))
            ] = pos2cls_cof(rddbase, num, basename, split)
        print(c)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        # return shape : tensor(Ws,Hs,C)
        im = loadimgsp(self.rddbase + "JPEGImages/" + self.namelist[idx] + ".jpg")
        return im, self.targetl[idx].reshape(-1).long()


import pickle
from loadimg import loadimgsp
from core import xml2clsconf


class YOLOcatPatchDataset(torch.utils.data.Dataset):
    def __init__(self, base, pklpath, txtpath):
        self.yolooutput = pickle.load(open(pklpath, 'rb'))
        self.base = base
        self.idlist = []
        # check if file exists
        # generate id list
        with open(txtpath) as f:
            rows = f.readlines()
        rows = list(map(lambda s: s.strip(), rows))
        for idx, (imgfile, _, _, _, _) in enumerate(self.yolooutput):
            if os.path.exists(self.base + '/Annotations/' + imgfile.split('.')[0] + '.xml') and imgfile.split('.')[
                0] in rows:
                self.idlist.append(idx)
        print(f'YOLOcat:{len(self)}')

    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, idx):
        image_file, out_box_xy, out_box_wh, out_box_confidence, out_box_cls_probs = self.yolooutput[self.idlist[idx]]
        out_box_cls_probs = out_box_cls_probs[:, :, :, :, :6]
        img = loadimgsp(self.base + 'JPEGImages/' + image_file)
        label_clsconf = xml2clsconf(self.base + '/Annotations/' + image_file.split('.')[0] + '.xml', split=13)
        return img, np.concatenate([out_box_cls_probs, out_box_confidence],
                                   axis=-1), label_clsconf, out_box_xy, out_box_wh, image_file

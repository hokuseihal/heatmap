import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from core import PIL2Tail


def noex(path):
    return os.path.splitext(path)[0]


# patch dataset


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
            raise ValueError
        for obj in root.iter("object"):
            xmlbox = obj.find("bndbox")
            if normalize:
                b = (
                    float(xmlbox.find("xmin").text) / w,
                    float(xmlbox.find("xmax").text) / w,
                    float(xmlbox.find("ymin").text) / h,
                    float(xmlbox.find("ymax").text) / h,
                )

            else:

                b = (
                    float(xmlbox.find("xmin").text),
                    float(xmlbox.find("xmax").text),
                    float(xmlbox.find("ymin").text),
                    float(xmlbox.find("ymax").text),
                )
            cls = classes.index(obj.find("name").text)
            bb.append((cls, b))
        return bb


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
                    int(ratio * len(os.listdir(self.negative_road_base))) :
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
                    for path in sorted(os.listdir(self.negative_base))[
                        : int((1 - ratio) * len(os.listdir(self.positive_base)))
                    ]
                ]
                + [
                    (self.negative_road_base + path, 0)
                    for path in os.listdir(self.negative_road_base)[
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
                    round(b[0] / w_d) : round(b[1] / w_d),
                    round(b[2] / h_d) : round(b[3] / h_d),
                    cls,
                ] = 1
            return self.transforms(im), r_tensor


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


def pos2cls_cof(base, imp, pos, sp):
    #TODO debug
    basename = imp.split(".")[0]
    bb = getbb(basename, xmlbase=base + "Annotations/")
    # TODO -1 is none class
    for cls, b in bb:
        r = torch.Tensor([cls, 1.0]) if isinbox(pos, sp, b) else torch.Tensor([-1, 0.0])
    return r


def num2postuple(num, sp):
    return num % sp[0], num // sp[0]


class RoadDamagePatchDataset(torch.utils.data.Dataset):
    def __init__(self, rddbase, patchbase, split, transforms=None):
        assert isinstance(split, tuple), "argment sp's type must be TUPLE."
        assert len(split) == 2, "argment sp's length must be 2."
        self.rddbase = rddbase
        self.patchbase = patchbase
        self.transforms = (
            Compose([Resize((768, 768)), PIL2Tail(*split, "torch")])
            if transforms is None
            else transforms
        )
        self.namelist = []
        self.num_cls = num_cls
        foll = ["Positive", "Negative", "Negative_road"]
        for fol in foll:
            for imp in os.listdir(patchbase + fol):
                num, basename = split_head_num(imp)
                if basename not in self.namelist:
                    self.namelist.append(basename)
        self.targetl = torch.zeros((len(self.namelist), *split, 2))

        # on each num ,get bbox info and restore if patch is positive
        for imp in os.listdir(patchbase + "Positive"):
            num, basename = split_head_num(imp)
            self.targetl[
                (self.namelist.index(basename), *num2postuple(num, split))
            ] = pos2cls_cof(rddbase, basename, num, split)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        # return shape : tensor(Ws,Hs,C)
        im = Image.open(self.rddbase + "JPEGImages/" + self.namelist[idx] + ".jpg")
        return self.transforms(im), self.targetl[idx, :, :]

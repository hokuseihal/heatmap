import csv
import torch.nn as nn
from core import PIL2Tail
from PIL import Image
import torch
import torch.nn.functional as F
from patch import PatchModel
from patch import patchmodel
from torchvision.transforms import Resize, Compose
from loadimg import loadimgsp

def round_sp(cor, s, m):
    return int(cor * s / m)


# load model
# readcsv
with open("y2rresult_001.csv") as cf:
    reader = csv.reader(cf)
    csvl = [row for row in reader]
#   for img in csv.row
pre = ""
NAME = 0
CLS = 1
PROB = 2
X1 = 3
Y1 = 4
X2 = 5
Y2 = 6
POSITIVE=1

limit = 0.5
#model = torch.load("patchmodel.pth", map_location="cpu")
model=patchmodel
model.eval()
for row in csvl:
    print(row[NAME])
    #       if different:heatmap=model(img)
    if not pre == row[NAME]:
        img = loadimgsp
        out=model(img)
        heatmap=F.softmax(out,dim=-1)
        heatmap=heatmap.reshape(6,6,2)
    #       if heatmap[img.bb]>limit:
    t1 = round_sp((int(row[X1]) + int(row[X2])) / 2, 6, 600)
    t2 = round_sp((int(row[Y1]) + int(row[Y2])) / 2, 6, 600)
    if (
        heatmap[
            round_sp((int(row[X1]) + int(row[X2])) / 2, 6, 600),
            round_sp((int(row[Y1]) + int(row[Y2])) / 2, 6, 600),
        ][POSITIVE]
        > limit
    ):
        #           out(row)
        print(
            f"{row[NAME]},{row[CLS]},{row[PROB]},{row[X1]},{row[Y1]},{row[X2]},{row[Y2]}"
        )
    pre = row[NAME]

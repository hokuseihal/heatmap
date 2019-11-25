import patch
import csv
from core import PIL2Tail
from PIL import Image
import torch
import torch.nn.functional as F


def round_sp(cor, s, m):
    return int(cor / m * s)


# load model
# readcsv
with open("result.csv") as cf:
    reader = csv.reader(cf)
    csvl = [row for row in reader]
s2t = PIL2Tail(6, 6, "torch")
#   for img in csv.row
pre = ""
NAME = 0
CLS = 1
PROB = 2
X1 = 3
Y1 = 4
X2 = 5
Y2 = 6

limit = 0.5
model = torch.load("patchmodel.pth")
model.eval()
for row in csvl:
    #       if different:heatmap=model(img)
    if not pre == row[NAME]:
        img = Image.open(row[NAME])
        img = s2t(img)
        heatmap = F.softmax(model(img), dim=-1)
    #       if heatmap[img.bb]>limit:
    if (
        heatmap[
            round_sp((row[X1] + row[X2]) / 2, 6, 600),
            round_sp((row[Y1] + row[Y2]) / 2, 6, 600),
        ]
        > limit
    ):
        #           out(row)
        print(
            f"{row[NAME]},{row[CLS]},{row[PROB]},{row[X1]},{row[Y1]},{row[X2]},{row[Y2]}"
        )
    pre = row[NAME]

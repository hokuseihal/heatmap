import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from itertools import product
import csv
from PIL import Image

classes = ["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44", "D30", "", "", ""]
valtxtpath = "All/ImageSets/Main/val.txt"
filename = "detect_ssd_mobile_test.csv"
dataroot = "All/Annotations/"
labelpath = "All/labels/"
imagefolder = "All/JPEGImages/"
n = 0


def readxml(r):
    global readclslist
    r_list = []
    try:
        tree = ET.parse(dataroot + "/" + r[:-1] + ".xml")

        root = tree.getroot()
        for child in root.iter("object"):
            for object in child:
                if "name" == object.tag:
                    cls = object.text
                if "bndbox" == object.tag:
                    r_list.append(
                        [int(xy.text) for xy in object] + [classes.index(cls)]
                    )
        return r_list
    except FileNotFoundError:
        print(r, ".xml not found.")


def readlabel(name):
    r = []
    label = labelpath+name.split('/')[-1].split('.')[0]+'.txt'
    # read csv
    data = np.loadtxt(label, delimiter=" ")
    # get image shape
    image = Image.open(name)
    imagew, imageh = image.size
    # append x0,y0,x1,y1,cls --!not to cal from yolo to voc
    if len(data.shape) == 1:
        data = [data]
    try:
        for row in data:
            x = row[1] * imagew
            y = row[2] * imageh
            w = row[3] * imagew
            h = row[4] * imageh
            r.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2, row[0]])
        # return
        return r
    except IndexError:
        print('Index Error')
        return []


def calioucls(gl, df):
    df = df[1]
    if not (gl[0] < df[5] and gl[2] > df[3] and gl[1] < df[6] and gl[3] > df[4]):
        return -1
    if gl[-1] != df[1]:
        return -2
    A = (gl[2] - gl[0]) * (gl[3] - gl[1])
    B = (df[5] - df[3]) * (df[6] - df[4])
    C = (min(df[5], gl[2]) - max(df[3], gl[0])) * (
        min(df[6], gl[3]) - max(df[4], gl[1])
    )
    iou = C / (A + B - C)
    return iou


def cal(gt_list, detect_df):
    global n
    gt_list = np.array(gt_list)
    tg = np.zeros((len(classes), len(gt_list)))
    td = np.zeros((len(classes), detect_df.shape[0]))
    for g, gt in enumerate(gt_list):
        for d, detect in enumerate(detect_df.iterrows()):
            # cal iou and same class?
            if not calioucls(gt, detect) >= 0.5:
                continue
            # cal path weights
            cls = int(gt[-1])
            t = tg[cls, g] == 0
            td[cls, d] = 1 if tg[cls, g] == 0 else -1
            tg[cls, g] = 1
            if td[cls, d] == -1:
                print("double!!", n)
                n += 1
            # return them
    fp = [0 for i in range(len(classes))]
    fn = [0 for i in range(len(classes))]
    for i, d in enumerate(((tg == 1).sum(axis=0) == 0)):
        if d:
            fn[int(gt_list[i, -1])] += 1
    for i, d in enumerate((((td == 1) | (td == -1)).sum(axis=0) == 0)):
        if d:
            fp[detect_df.iloc[i, 1]] += 1
    tp = (tg == 1).sum(axis=1)
    return tp, fp, fn


detectdf = pd.read_csv(filename, sep=",", header=None)
with open(valtxtpath) as f:
    vallines = f.readlines()

tp = np.zeros((len(classes)))
fp = np.zeros((len(classes)))
fn = np.zeros((len(classes)))
for name in vallines:
    try:
        name = imagefolder + name.split(' ')[0].strip() + ".jpg"
        # gt=readxml(name)
        gt = readlabel(name)
        dect = detectdf[detectdf[0] == name.split(".")[0]]
        if len(gt) == 0:
            print(name[:-1], "not found.")
        t1, t2, t3 = cal(gt, dect)
        tp += t1
        fp += t2
        fn += t3
    except OSError:
        print(name, "not found")

print("prediction:", tp / (tp + fp))
print("recall:", tp / (tp + fn))
print("END_PREDICTION_RECALL")

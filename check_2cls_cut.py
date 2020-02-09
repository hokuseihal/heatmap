import csv
from core import checkRDD
from loadimg import loadimgsp
import torch
from math import ceil,floor
import numpy as np
def check_2cls_cut(model,device,csvname='01test.csv',thresh=0.5):
    prmap=np.zeros((2,2))

    with open(csvname) as f:
        lines=[i for i in  csv.reader((line.replace('\0','') for line in f) ) if 0.01<float(i[2])<0.5]

    for detect in lines[:200]:
        imgpath,cls,prob,x0,y0,x1,y1=detect
        cls, prob, x0, y0, x1, y1=float(cls),float(prob),float(x0),float(y0),float(x1),float(y1)
        label=int(checkRDD(imgpath,(cls,(x0,y0,x1,y1))))
        img = loadimgsp(imgpath)
        img = img.to(device)
        model.eval()
        with torch.no_grad():
            out = model(img)
        out = out.view(6, 6, 2)
        out = torch.softmax(out, dim=-1)
        x0, y0, x1, y1 = max(0,floor(x0 / 100)), max(0,floor(y0 / 100)), min(6,ceil(x1 / 100)), min(6,ceil(y1 / 100))
        bi_cls=int(thresh < out[x0:x1, y0:y1, 1].max())
        prmap[label,bi_cls]+=1
    precision=np.diag(prmap)/prmap.sum(axis=0)
    recall=np.diag(prmap)/prmap.sum(axis=-1)
    f1=1/(1/precision+1/recall)
    print(f'Precision:{precision},Recall:{recall}')
    return np.nanmean(f1)
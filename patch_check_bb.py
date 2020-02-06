from core import readxml
import pickle
from math import ceil,floor
from loadimg import loadimgsp
import torch
import numpy as np
def check_bb(patchmodel,device):

    thresh=0.5
    tp=0
    fp=0
    t=0
    yolodic={}
    ben_y2=np.zeros((2,3))
    with open('yolo_check_bb.txt') as ycbf:
        lines=ycbf.readlines()
        lines= [line.strip().split(' ') for line in lines]
        for line in lines:
            yolodic[line[0]]=list(map(float,line[1:]))
    with open('/home/hokusei/src/mydarknet/test.txt') as f:
        lines=f.readlines()
    for line in lines:
        result=[]
        line=line.strip()
        #print(line,end='')
        gt=readxml(line)
        t+=len(gt)
        img = loadimgsp(line)
        img = img.to(device)
        patchmodel.eval()
        with torch.no_grad():
            out = patchmodel(img)
        out = out.view(6, 6, 2)
        out = torch.softmax(out, dim=-1)
        try:
            yolodic[line]
        except KeyError:
            yolodic[line]=np.zeros(len(gt))
        for idx,(x0,y0,x1,y1,cls) in enumerate(gt):
            if cls==6 or cls==7:
                result.append(2)
                continue
            x0,y0,x1,y1=floor(x0/100),floor(y0/100),ceil(x1/100),ceil(y1/100)
            if thresh < out[x0:x1,y0:y1,1].max():
                #print(' 1',end='')
                tp+=1
                result.append(True)
            else:
                #print(' 0', end='')
                fp+=1
                result.append(False)
        assert len(yolodic[line])==len(result)
        for i in range(len(result)):
            ben_y2[int(yolodic[line][i]),int(result[i])]+=1
    print()
    return tp/(tp+fp)

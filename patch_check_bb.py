from core import readxml
import pickle
from math import ceil,floor
from loadimg import loadimgsp
import torch

def check_bb(patchmodel,device):

    thresh=0.5
    tp=0
    fp=0

    with open('/home/hokusei/src/mydarknet/test.txt') as f:
        lines=f.readlines()
    for line in lines:
        line=line.strip()
        #print(line,end='')
        gt=readxml(line)
        img = loadimgsp(line)
        img = img.to(device)
        patchmodel.eval()
        with torch.no_grad():
            out = patchmodel(img)
        out = out.view(6, 6, 2)
        out = torch.softmax(out, dim=-1)
        for x0,y0,x1,y1,cls in gt:
            if cls==6 or cls==7:
                continue
            x0,y0,x1,y1=floor(x0/100),floor(y0/100),ceil(x1/100),ceil(y1/100)
            if thresh < out[x0:x1,y0:y1,1].max():
                #print(' 1',end='')
                tp+=1
            else:
                #print(' 0', end='')
                fp+=1
    print()
    return tp/(tp+fp)

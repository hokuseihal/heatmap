from PIL import Image
import numpy as np
import cv2
import torch
from torchvision.transforms import ToTensor,Resize
def loadimgsp(imp,sp=6):
#load ,split and save img
    img = cv2.imread(imp)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    size = img.shape[0]/sp
    v_size = int(img.shape[0] // size * size)
    h_size = int(img.shape[1] // size * size)
    img = img[:v_size, :h_size]

    v_split = img.shape[0] // size
    h_split = img.shape[1] // size
    out_img = []
    [out_img.extend(np.hsplit(h_img, h_split)) for h_img in np.vsplit(img, v_split)]
    for i in range(len(out_img)):
        out_img[i]=ToTensor()(Resize((128,128))(Image.fromarray(out_img[i])))
    return torch.stack(out_img)

import pickle
class Load2cls:
    def __init__(self,pklp):
        with open(pklp, mode='rb') as f:
            self.dic= pickle.load(f)
    def get(self,imp):
        return self.dic[imp]

loadimgsp('All/JPEGImages/Adachi_20170914152124.jpg')
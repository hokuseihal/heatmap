from PIL import Image, ImageDraw
from core import PIL2Tail
from torchvision.transforms import Resize
import cv2
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from mein import putboxes, del_trimming, canny, kmeans, Aspect
import torch
from patch import PatchModel
import torch.nn.functional as F
from loadimg import loadimgsp
root = os.environ["HOME"] + "/src/RoadDamageDataset/All/"
labelroot = root + "labels/"
imageroot = root + "JPEGImages/"
pmodel = torch.load("patchmodel.pth", "cpu")
from patch import patchmodel
pmodel=patchmodel.to('cpu')
patch = True
for i in os.listdir(imageroot):
    try:
        img = cv2.imread("All/JPEGImages/Adachi_20170907134630.jpg")
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        assert img.any()
        boxes = np.loadtxt(
            labelroot + i.replace("jpg", "txt"), delimiter=" ", dtype=np.float32
        )
        # convimg = kmeans(img,boxes)
        # dimg = del_trimming(convimg, boxes)
        if not putboxes(boxes, img):
            continue
        print()
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax2 = fig.add_subplot(1, 2, 2)
        if patch:
            pimg = Image.open("All/JPEGImages/Adachi_20170907134630.jpg")
            draw = ImageDraw.Draw(pimg)
            img = loadimgsp("All/JPEGImages/Adachi_20170907134630.jpg")
            out = pmodel(img)
            heatmap = F.softmax(out, dim=-1)
            heatmap=heatmap.reshape(6, 6, 2)
            print(heatmap.argmax(dim=-1))
            for i in range(6):
                for j in range(6):
                    if heatmap[i, j,1] > 0.5:
                        draw.rectangle(
                            (100 * i, 100 * j, 100 * (i + 1), 100 * (j + 1))
                        )

            ax2.imshow(pimg)
        plt.show()
    except KeyboardInterrupt:
        exit(0)
    except:
        print(f"error show {i}")
        import traceback

        traceback.print_exc()

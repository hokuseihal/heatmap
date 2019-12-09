import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw
from loadimg import loadimgsp
from mein import putboxes
from patch import patchmodel

root = os.environ["HOME"] + "/src/RoadDamageDataset/All/"
labelroot = root + "labels/"
imageroot = root + "JPEGImages/"
#pmodel = torch.load("patchmodel.pth", "cpu")
pmodel = patchmodel.to("cpu")
patch = True
for i in os.listdir(imageroot):
    try:
        img = cv2.imread(imageroot + i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            pimg = Image.open(imageroot + i)
            draw = ImageDraw.Draw(pimg)
            img = loadimgsp(imageroot + i)
            out = pmodel(img)
            heatmap = F.softmax(out, dim=-1)
            heatmap = heatmap.reshape(6, 6, 2)
            print(heatmap.argmax(dim=-1))
            for i in range(6):
                for j in range(6):
                    if heatmap[i, j, 1] > 0.5:
                        draw.rectangle(
                            (100 * j, 100 * i, 100 * (j + 1), 100 * (i + 1)),
                            fill=(0, 0, 0),
                        )

            ax2.imshow(pimg)
        plt.show()
    except KeyboardInterrupt:
        exit(0)
    except:
        print(f"error show {i}")
        import traceback

        traceback.print_exc()

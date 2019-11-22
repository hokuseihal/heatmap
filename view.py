import cv2
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from mein import putboxes, del_trimming,canny,kmeans,Aspect

root = os.environ["HOME"] + "/src/RoadDamageDataset/All/"
labelroot = root + "labels/"
imageroot = root + "JPEGImages/"
for i in os.listdir(imageroot):
    try:
        img = cv2.imread(imageroot + i)
        assert img.any()
        boxes = np.loadtxt(
            labelroot + i.replace("jpg", "txt"), delimiter=" ", dtype=np.float32
        )
        #convimg = kmeans(img,boxes)
        #dimg = del_trimming(convimg, boxes)
        if not putboxes(boxes, img):
            continue
        print()
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img)
        plt.show()
    except KeyboardInterrupt:
        exit(0)
    except:
        print(f"error show {i}")
        import traceback
        traceback.print_exc()

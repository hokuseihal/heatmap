import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np


def y2v(box, width, height):
    x1 = int((box[1] - box[3] / 2) * width)
    y1 = int((box[2] - box[4] / 2) * height)
    x2 = int((box[1] + box[3] / 2) * width)
    y2 = int((box[2] + box[4] / 2) * height)
    return x1, y1, x2, y2


def del_trimming(img, boxes):
    for box in boxes:
        if int(box[0])<=5:
            x1, y1, x2, y2 = y2v(box, 600, 600)
            dimg[y1:y2, x1:x2] = result[y1:y2, x1:x2]
    return dimg


img = cv2.imread("../RoadDamageDataset/All/JPEGImages/Adachi_20170921155217.jpg")
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
imgY = imgYUV[:, :, 0]

result = cv2.Canny(imgY, 200, 300)
fig = plt.figure(figsize=(10, 5))
boxes = np.loadtxt(
    "../RoadDamageDataset/All/labels/Adachi_20170921155217.txt",
    delimiter=" ",
    dtype=np.float32,
)
boxes=boxes.reshape(-1,5)
dimg = np.zeros_like(result)
dimg = del_trimming(result, boxes)
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img)
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(dimg)
plt.show()

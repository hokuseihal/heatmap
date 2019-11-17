import cv2
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

root = os.environ["HOME"] + "/src/RoadDamageDataset/All/"
labelroot = root + "labels/"
imageroot = root + "JPEGImages/"
imagesaveroot=imageroot+'kmeans/'
COLORS = np.random.uniform(0, 255, size=(8, 3))
os.makedirs(imagesaveroot,exist_ok=True)
for index,i in enumerate(os.listdir(imageroot)):
    print({index})
    try:
        img = cv2.imread(imageroot + i)
        s=img.shape
        assert img.any()
                
        img = img.reshape((-1, 3)).astype(np.float32)
        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(
            img, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((s))
        cv2.imwrite(imagesaveroot+i,res2)
    except FileNotFoundError:
        print(f"error show {i}")


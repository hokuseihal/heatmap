import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch


class RDD:
    def __init__(self):
        pass

    def __len__():
        pass

    def __getitem__(idx):
        pass


COLORS = np.random.uniform(0, 255, size=(10, 3))
classes = ["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44", "D30"]


class Aspect:
    def __init__(self, gimg):
        self.n = 128
        self.gimg = gimg
        self.r = np.zeros((self.n, self.n))

        # makemap
        self.map = cv2.resize(gimg, (self.n, self.n)) > 128
        cv2.imwrite("stack.jpg", self.map * 255)

    def count(self, x, y, d):
        if not (0 < x < self.n) or not (0 < y < self.n) or self.map[x, y] == 0:
            return 0
        return 1 + self.count(x + int(d / 2), y + d % 2, d)

    def aspect(self, i, j):
        UP = -1
        DOWN = 1
        LEFT = -2
        RIGHT = 2
        return (
            (self.count(i, j - 1, UP) + self.count(i, j + 1, DOWN)),
            (self.count(i - 1, j, LEFT) + self.count(i + 1, j, RIGHT)),
        )

    def makeaspectmap(self, k=10):
        # inner funca
        for i in range(16):
            for j in range(16):
                self.r[i, j] = max(
                    int(np.linalg.norm(self.aspect(i, j)) - np.sqrt(2) * k), 255
                )
        # return map
        return cv2.resize(self.r, self.gimg.shape)


def y2v(box, width, height):
    x1 = int((box[1] - box[3] / 2) * width)
    y1 = int((box[2] - box[4] / 2) * height)
    x2 = int((box[1] + box[3] / 2) * width)
    y2 = int((box[2] + box[4] / 2) * height)
    return x1, y1, x2, y2


def del_trimming(img, boxes, cut=False):
    dimg = np.zeros_like(img) if not cut else np.ones_like(img) * -1
    boxes = boxes.reshape(-1, 5)
    for box in boxes:
        if int(box[0]) <= 5:
            x1, y1, x2, y2 = y2v(box, 600, 600)
            dimg[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return dimg


def trimminglist(img, boxes):
    r = []
    boxes = boxes.reshape(-1, 5)
    for box in boxes:
        if int(box[0]) <= 5:
            x1, y1, x2, y2 = y2v(box, 600, 600)
            r.append(img[y1:y2, x1:x2])
    return r


def putboxes(boxes, img, wantcls=None):
    boxes = boxes.reshape(-1, 5)
    if (wantcls is not None) and (not wantcls in boxes[:, 0]):
        return False
    for box in boxes:
        cls = int(box[0])
        color = COLORS[int(cls)]
        x1, y1, x2, y2 = y2v(box, 600, 600)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            classes[cls],
            (x1 + 10, y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return True


def canny(img, grey=False):
    imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    imgY = imgYUV[:, :, 0]
    cnyimg = cv2.Canny(imgY, 200, 300)
    if grey:
        return cv2.cvtColor(cnyimg, cv2.COLOR_YUV2GRAY_420)
    return cnyimg


def kmeans(img, boxes=None, black=False):

    assert img.any()
    s = img.shape
    if boxes is None:
        oriimg = img
        img = img.reshape((-1, 3)).astype(np.float32)
    else:
        limg = trimminglist(img, boxes)
        # TODO what to do?
        img = limg[0]
        s = img.shape
        img = img.reshape((-1, 3)).astype(np.float32)
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        img, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(s)
    if black:
        b = center.sum(1)
        b = b.argmin()
        return np.uint8(res2 == center[b]) * 255
    return res2

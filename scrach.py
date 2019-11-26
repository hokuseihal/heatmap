import numpy as np
import torch
from PIL import Image
import cv2

im=cv2.imread()
w_d = im.size[0] // w_s
h_d = im.size[0] // h_s

im = np.array(im.rotate(90))
im = im.reshape(
    im.shape[0] // w_d,
    w_d,
    im.shape[1] // h_d,
    h_d,
    im.shape[2],
).swapaxes(1, 2)
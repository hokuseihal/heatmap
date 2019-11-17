import os
import cv2
import numpy as np
from mein import Aspect, canny

imagefolder = ""
asmapfolder = imagefolder + "asmap/"

for i in os.listdir(imagefolder):
    img = cv2.imread(imagefolder + i)
    assert img
    greyimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cannyimg = canny(img, grey=True)
    aspectmap = Aspect(greyimg).makeaspectmap()
    assert greyimg.shape == cannyimg.shape == aspectmap.shape
    cv2.imwrite(asmapfolder + i, [greyimg, cannyimg, aspectmap])

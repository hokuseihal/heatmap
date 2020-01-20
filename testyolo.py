import os
import subprocess
from cal_score5 import precision_recall
backuppath=''

for w in os.listdir(backuppath):
    call=['./darknet','rdd','cg/yolov2-rdd.cfg',w,'test.txt']
    subprocess.call(call)
    precision_recall(csvfilename='result.csv')
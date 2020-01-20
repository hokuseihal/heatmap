import os
import subprocess
from cal_score5 import precision_recall
backuppath='/home/wadanaoki/src/mydarknet/backup'

for w in os.listdir(backuppath):
    call=['/home/wadanaoki/src/mydarknet/darknet','rdd','cg/yolov2-rdd.cfg',w,'test.txt']
    subprocess.call(call)
    precision_recall(csvfilename='result.csv')
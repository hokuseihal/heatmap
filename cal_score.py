import csv
import xml.etree.ElementTree as ET
import numpy as np
class NotinClassError(Exception):
    pass
filename='csv/detection_ssd_mobile_trainval.csv'
dataroot='/home/popo/All/Annotations'
classes = ['D00', 'D01', 'D10', 'D11', 'D20','D40', 'D43', 'D44','D30']
tpfn_tp=np.zeros(len(classes))
tpfp_tp=np.zeros(len(classes))
tpfn=np.zeros(len(classes))
tpfp=np.zeros(len(classes))
count=np.zeros(len(classes))
oops=0
filelist=[]
with open(filename) as f:
    reader=csv.reader(f)
    for row in reader:
        fpflag=False
        if row[0] in filelist:
            flag=False
        else:
            filelist.append(row[0])
            flag=True
        row[1]=int(row[1])
        row[3]=int(row[3])
        row[4]=int(row[4])
        row[5]=int(row[5])
        row[6]=int(row[6])
        tpfp[row[1]]+=1

        try:
            tree=ET.parse(dataroot+'/'+row[0]+'.xml')

            root=tree.getroot()
            for child in root.iter('object'):
                for object in child:
                    if 'name'==object.tag:
                        cls=object.text
                    if 'bndbox'==object.tag:
                        xml=[int(xy.text) for xy in object]
                        if flag:
                            tpfn[classes.index(cls)]+=1
                        #if not collision then skip
                        if not (row[3]<xml[2] and xml[0]<row[5]  and row[4]<xml[3] and xml[1]<row[6]) or cls!=classes[row[1]]:
                            continue
                        A=(xml[2]-xml[0])*(xml[3]-xml[1])
                        B=(row[5]-row[3])*(row[6]-row[4])
                        C=(min(row[5],xml[2])-max(row[3],xml[0]))*(min(row[6],xml[3])-max(row[4],xml[1]))
                        iou=C/(A+B-C)
                        if iou>0.5:
                            tpfn_tp[classes.index(cls)]+=1
                            fpflag=True
                            if not flag:
                                tpfn[classes.index(cls)]+=1
                                count[row[1]]+=1

        except FileNotFoundError:
            print('Oops! '+dataroot+'/'+row[0]+'.xml is not Found')
            tpfp[row[1]]-=1
            oops+=1
        if fpflag:tpfp_tp[row[1]]+=1

with open(filename) as f:
    print(sum(tpfp),len(f.readlines())-oops)
#TDDO eveything must be subbed double_count
print('precision',tpfp_tp/tpfp)
print('recall',tpfn_tp/tpfn)
print("tpfn:",tpfn," tpfp:",tpfp," tpfn_tp:",tpfn_tp,"tpfp_tp",tpfp_tp)
print('oops',oops)
print('double count',tpfp_tp-tpfn_tp)
import numpy as np
count=np.zeros(8)

with open('/home/hokusei/src/mydarknet/test.txt') as f:
    for s in f.readlines():
        with open('All/labels/'+s.strip().split('/')[-1].split('.')[-2]+'.txt') as ff:
            for line in ff.readlines():
                c,_,_,_,_=line.split(' ')
                count[int(c)]+=1
print(count)

with open('yolo_check_bb.txt') as f:
    lines=[l.split(' ') for l in f.readlines()]
s=sum([len(l) for l in lines])
print(s-len(lines))
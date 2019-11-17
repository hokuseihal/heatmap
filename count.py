import os

limit = 25
labelf = "All/labels/"
imagef='All/JPEGImages/'
count = [0 for i in range(8)]
for i in os.listdir(labelf):
    with open(labelf + i) as f:
        for l in f.readlines():
            if count[int(l[0])] < limit:
                count[int(l[0])] += 1
                print(imagef +i.split('.')[0]+'.jpg' )


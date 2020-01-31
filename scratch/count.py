import os
labelbase='home/wadanaoki/src/mydarknet/img'
count=[0,0,0,0,0,0,0,0]
for i in os.listdir(labelbase):
    if 'txt' in i:
        with open(labelbase+'/'+i) as f:
            lines=[i.split(' ') for i in f.readlines()]
        for line in lines:
            count[int(line[0])]+=1
print(count)

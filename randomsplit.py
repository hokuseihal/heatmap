import random
traintxt='/home/hokusei/src/mydarknet/train.txt'
N=2
with open(traintxt) as f:
    lines=f.readlines()
num=len(lines)

random.shuffle(lines)
for i in range(N):
    with open(traintxt.split('.')[0]+str(i)+'.txt','w') as f:
        for l in lines[i*num//N:(i+1)*num//N]:
            f.write('%s'%l)

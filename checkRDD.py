#check bounding box is ok (same class ,iou is greater than thresh)
import torch
import torch.nn as nn
from dataset import getbb,YOLOOutputDataset as dataset
import os
from models import ImgPackModel as Model
from core import train,test,SoftmaxFocalLoss

def main():
    batchsize = 72
    num_epoch = 252
    model_save_path='imgpackmodel.pth'
    csvpath='y2rresult_001.csv'


    train_dataset = dataset('All/',csvpath, 'All/ImageSets/Main/train.txt')
    test_dataset =dataset('All/',csvpath, 'All/ImageSets/Main/val.txt')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, shuffle=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    print(device)
    model=Model().to(device)
    #if os.path.exists(model_save_path):
    #    model.load_state_dict(torch.load(model_save_path))
    #    print('load weight')
    optimizer = torch.optim.Adam(model.parameters())
    lossf=SoftmaxFocalLoss

    for e in range(num_epoch):
        #train
        model.train()
        losslist=[]
        # log_interval = len(train_loader)
        log_interval = 3
        for batch_idx, (img,splittedimg,bbox,mappedbox, target) in enumerate(train_loader):
            img,splittedimg,bbox,mappedbox, target = img.to(device), splittedimg.to(device),bbox.to(device),target.to(device)
            optimizer.zero_grad()
            output = model(img,splittedimg,bbox,mappedbox)
            loss = lossf(output, target)
            loss.backward()
            optimizer.step()
            losslist.append(loss.item())
            if (batch_idx + 1) % log_interval == 0:
                print(f'Train Epoch: {e} [{batch_idx*batchsize}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {np.mean(losslist):.6f}')
                losslist=[]
        #test
        torch.save(model.state_dict(), model_save_path)



if __name__=='__main__':
    main()
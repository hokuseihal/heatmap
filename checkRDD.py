# check bounding box is ok (same class ,iou is greater than thresh)
import torch
import torch.nn as nn
from dataset import getbb, YOLOOutputDataset as dataset
import os
from models import ImgPackModel as Model
from core import train, test, SoftmaxFocalLoss
import numpy as np
from core import prmap, patchaccf
from torch.utils.tensorboard import SummaryWriter
from cal_score5 import precision_recall


def main():
    writer = SummaryWriter()
    batchsize = 16
    num_epoch = 1
    model_save_path = 'imgpackmodel.pth'
    testcsv = 'result_test_001.csv'
    traincsv = 'result_train_001.csv'
    traincsv='result0_test.csv'

    train_dataset = dataset('All/', traincsv)
    test_dataset = dataset('All/', testcsv)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, shuffle=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    print(device)
    model = Model().to(device)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print('load main weight')
    optimizer = torch.optim.Adam(model.parameters())
    lossf = SoftmaxFocalLoss(gammma=2)
    num_epoch = num_epoch * len(train_dataset) // batchsize
    ##test start
    z = 0
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    g=0
    for batch_idx, (img, splittedimg, mappedbox, bbox, target, idx) in enumerate(train_loader):
        g+=(target.bool()).sum()
        d+=(target.bool().logical_not()).sum()
        e+=(bbox[1]>0.5).sum()
        f+=(bbox[1]<0.5).sum()
        b+=(target.bool() & (bbox[1] > 0.5)).sum()
        z += (target.bool().logical_not() & (bbox[1]>0.5)).sum()
        c+=(target.bool() & (bbox[1]<0.5)).sum()
        a+=(target.bool().logical_not() & (bbox[1]>0.5).logical_not()).sum()
    print(g,d,e,f,b,z,c,a)
    exit(0)
    ##test end
    for e in range(num_epoch):
        # train
        model.train()
        # log_interval = len(train_loader)
        log_interval = 8
        losslist = []
        for batch_idx, (img, splittedimg, mappedbox, bbox, target, idx) in enumerate(train_loader):
            img, splittedimg, mappedbox, target = img.to(device), splittedimg.to(device), mappedbox.to(
                device), target.to(device)
            optimizer.zero_grad()
            output = model(img, splittedimg, bbox, mappedbox)
            loss = lossf(output, target)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss:Train', loss.item(), e)
            losslist.append(loss.item())
            if (batch_idx + 1) % log_interval == 0:
                print(f'Train Epoch: {e} [{batch_idx * batchsize}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {np.mean(losslist):.6f}')
                break
        # test
        losslist = []
        correct = 0
        rmap = 0
        thresh = .5
        oklist = np.zeros(len(test_dataset))
        for batch_idx, (img, splittedimg, mappedbox, bbox, target, idx) in enumerate(test_loader):
            with torch.no_grad():
                img, splittedimg, mappedbox, target = img.to(device), splittedimg.to(device), mappedbox.to(
                    device), target.to(device)
                optimizer.zero_grad()
                output = model(img, splittedimg, bbox, mappedbox)
                # cal accuracy
                # cal prmap
                loss = lossf(output, target)
                writer.add_scalar('Loss:Test', loss.item(), e)
                losslist.append(loss.item())
                pred = output.argmax(dim=-1, keepdim=True)
                correct += patchaccf(target, pred)
                rmap += prmap(target, output)
                biggerprob=torch.zeros_like(bbox[1],dtype=torch.bool)
                #biggerprob = bbox[1] > thresh
                oklist[idx] = ((pred.view(-1).cpu() == 1) | biggerprob).numpy()

        print(f'Test Epoch: {e} [{batch_idx}/{len(test_loader)} ({100.0 * batch_idx / len(test_loader):.0f}%)]\tLoss: {np.mean(losslist):.6f}')
        print(f'precision:{rmap.diag() / rmap.sum(dim=-1)}\nrecall:{rmap.diag() / rmap.sum(dim=0)}')
        precision_recall(testcsv, oklist)
        # exit(1)
        torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    main()

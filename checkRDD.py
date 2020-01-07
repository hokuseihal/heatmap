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
from cal_score3 import precision_recall


def main():
    writer = SummaryWriter()
    batchsize = 16
    num_epoch = 1
    model_save_path = 'imgpackmodel.pth'
    testcsv='result_test_001.csv'
    traincsv='result_train_001.csv'

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
        print('load weight')
    optimizer = torch.optim.Adam(model.parameters())
    lossf = SoftmaxFocalLoss()
    num_epoch = num_epoch * len(train_dataset) // batchsize
    for e in range(num_epoch):
        # train
        model.train()
        #log_interval = len(train_loader)
        log_interval = 32
        losslist = []
        #for batch_idx, (img,splittedimg,mappedbox,bbox, target,idx) in enumerate(train_loader):
        #   img,splittedimg,mappedbox, target = img.to(device), splittedimg.to(device),mappedbox.to(device),target.to(device)
        #   optimizer.zero_grad()
        #   output = model(img,splittedimg,bbox,mappedbox)
        #   loss = lossf(output, target)
        #   loss.backward()
        #   optimizer.step()
        #   writer.add_scalar('Loss:Train',loss.item(),e)
        #   losslist.append(loss.item())
        #   if (batch_idx + 1) % log_interval == 0:
        #       print(f'Train Epoch: {e} [{batch_idx*batchsize}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {np.mean(losslist):.6f}')
        #       break
        # test
        losslist = []
        correct = 0
        rmap = 0
        thresh = .5
        oklist = np.zeros(len(test_dataset))
        for batch_idx, (img, splittedimg, mappedbox, bbox, target, idx) in enumerate(test_loader):
            log_interval=len(test_loader)
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
                biggerprob = bbox[1] > thresh
                oklist[idx]=((pred.view(-1).cpu()==1) | biggerprob).numpy()
                if (batch_idx + 1) % log_interval == 0:
                    print(f'Test Epoch: {e} [{batch_idx}/{len(train_loader)} ({100.0 * batch_idx / len(test_loader):.0f}%)]\tLoss: {np.mean(losslist):.6f}')
                    losslist = []
                    print(f'precision:{rmap.diag() / rmap.sum(dim=-1)}\nrecall:{rmap.diag() / rmap.sum(dim=0)}')
                    precision_recall(testcsv,oklist)
        exit(1)
        torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    main()

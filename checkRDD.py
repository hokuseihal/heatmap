# check bounding box is ok (same class ,iou is greater than thresh)
import torch
import torch.nn as nn
from dataset import getbb, YOLOOutputDataset as dataset
import os
from models import ImgPackModel2 as Model
from core import train, test, SoftmaxFocalLoss
import numpy as np
from core import prmap, patchaccf
from torch.utils.tensorboard import SummaryWriter
from cal_score5 import precision_recall, tester
import time
from cal_score5 import Precison_Recall_Teseter as PRT

def main():
    writer = SummaryWriter()
    batchsize = 64
    num_epoch = 100
    model_save_path = 'imgpackmodel.pth'
    traincsv = 'detect_ssd_inception_train.csv'
    testcsv = 'detect_ssd_inception_test.csv'
    #traincsv = '01test.csv'
    probthresh = .01
    train_dataset = dataset('All/', traincsv, prob_thresh=probthresh)
    test_dataset = dataset('All/', testcsv, prob_thresh=probthresh)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, shuffle=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = 'cpu'
    print(device)
    model = Model().to(device)
    #if os.path.exists(model_save_path):
    #    model.load_state_dict(torch.load(model_save_path))
    #    print('load main weight')
    optimizer = torch.optim.Adam(model.parameters())
    objlossf = SoftmaxFocalLoss(gammma=2)
    prtester=PRT(testcsv=testcsv,probthresh=probthresh)
    #test start
    #z = 0
    #a = 0
    #b = 0
    #c = 0
    #d = 0
    #e = 0
    #f = 0
    #g = 0
    #for batch_idx, (img, splittedimg, mappedbox, bbox, target, idx) in enumerate(train_loader):
    #   g += (target.bool()).sum()
    #   d += (target.bool().logical_not()).sum()
    #   e += (bbox[1] > 0.5).sum()
    #   f += (bbox[1] < 0.5).sum()
    #   b += (target.bool() & (bbox[1] > 0.5)).sum()
    #   z += (target.bool().logical_not() & (bbox[1] > 0.5)).sum()
    #   c += (target.bool() & (bbox[1] < 0.5)).sum()
    #   a += (target.bool().logical_not() & (bbox[1] > 0.5).logical_not()).sum()
    #print(e, f, g, d, b, z, c, a)
    #print(np.mean(train_dataset.c))
    #exit(0)
    #test end
    for e in range(num_epoch):
        ## train
        model.train()
        # log_interval = len(train_loader)
        log_interval = 16
        losslist = []
        for batch_idx, (img, splittedimg, mappedbox, bbox, objtarget, idx) in enumerate(train_loader):
            img, splittedimg, mappedbox, objtarget = img.to(device), splittedimg.to(device), mappedbox.to(
                device), objtarget.to(device)
            optimizer.zero_grad()
            out_obj = model(img, splittedimg, bbox, mappedbox)
            loss = objlossf(out_obj, objtarget)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss:Train', loss.item(), e)
            losslist.append(loss.item())
            if (batch_idx + 1) % log_interval == 0:
                print(f'Train Epoch: {e} [{batch_idx * batchsize}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {np.mean(losslist):.6f}')
                losslist = []
                # break
        # test
        correct = 0
        rmap = 0
        thresh = .5
        oklist = np.zeros(len(test_dataset))
        losslist = []
        model.eval()
        for batch_idx, (img, splittedimg, mappedbox, bbox, objtarget, idx) in enumerate(test_loader):
            img, splittedimg, mappedbox, objtarget = img.to(device), splittedimg.to(device), mappedbox.to(
                device), objtarget.to(device)
            with torch.no_grad():
                out_obj = model(img, splittedimg, bbox, mappedbox)
                loss = objlossf(out_obj, objtarget)
                writer.add_scalar('Loss:Test', loss.item(), e)
                losslist.append(loss.item())
                pred = out_obj.argmax(dim=-1, keepdim=True)
                correct += patchaccf(objtarget, pred)
                rmap += prmap(objtarget, out_obj)
                oklist[idx] = (out_obj[:,1]).detach().cpu().numpy()

        prtester.precisoin_recall_test(oklist=oklist)
        print(f'Test Epoch: {e} [{batch_idx}/{len(test_loader)} ({100.0 * batch_idx / len(test_loader):.0f}%)]\tLoss: {np.mean(losslist):.6f}')
        print(f'precision:{rmap.diag() / rmap.sum(dim=0)}\nrecall:{rmap.diag() / rmap.sum(dim=-1)}')
        # exit(1)
        torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    main()

import torch
import torch.nn.functional as F


class PIL2Tail(object):
    def __init__(self, w_s, h_s, r="numpy"):
        self.w_s = w_s
        self.h_s = h_s
        self.r = r

    def __call__(self, im):
        import numpy as np
        import torch

        self.w_d = im.size[0] // self.w_s
        self.h_d = im.size[0] // self.h_s

        im = np.array(im.rotate(90))
        im = im.reshape(
            im.shape[0] // self.w_d,
            self.w_d,
            im.shape[1] // self.h_d,
            self.h_d,
            im.shape[2],
        ).swapaxes(1, 2)
        if self.r == "numpy":
            # (NW,NH,W,H,C)
            return im
        elif self.r == "torch":
            # (NW,NH,C,H,W)
            return torch.from_numpy(im).permute(0, 1, 4, 3, 2)


def train(model, device, train_loader, lossf, optimizer, epoch, log_interval=1):
    model.train()
    tloss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float32), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossf(output, target)
        loss.backward()
        optimizer.step()
        tloss = tloss + loss.item()
        if (batch_idx + 1) % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    tloss / log_interval,
                )
            )
            tloss = 0


def test(model, device, test_loader, lossf, accf, mode="correct"):
    # accf:input:*labels,*labels
    #    :return:number of TP
    model.eval()
    test_loss = 0
    correct = 0
    tp_fp_tn_fn = torch.zeros(4, 6)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float32), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += lossf(output, target).item()
            # get the index of the max log-probability
            if mode == "correct":
                pred = output.argmax(dim=-1, keepdim=True)
                correct += accf(target, pred)
            elif mode == "tp_fp_tn_fn":
                tp_fp_tn_fn += accf(target, output)
            else:
                assert False, "Set correct mode"

    if mode == "correct":
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss/len(test_loader),
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    elif mode == "tp_fp_tn_fn":
        tp, fp, tn, fn = tp_fp_tn_fn
        print(
            f"Test set:Aberage loss:{test_loss:.4f}\n,tp:{tp}\n,fp:{fp}\n,tn:{tn}\n,fn:{fn}\n,precision:{tp/(tp+fp)}\n,recall:{tp/(tp+tn)}"
        )
    else:
        assert False, "Set correct mode"

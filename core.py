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

        self.w_d=im.size[0]//self.w_s
        self.h_d=im.size[0]//self.h_s

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
        data, target = data.to(device,dtype=torch.float32), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossf(output, target)
        loss.backward()
        optimizer.step()
        tloss = tloss + loss.item()
        if (batch_idx+1) % log_interval == 0:
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


def test(model, device, test_loader, lossf, accf):
    # accf:input:*labels,*labels
    #    :return:number of TP
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += lossf(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=-1, keepdim=True)
            correct += accf(target, pred)

    test_loss /= len(test_loader)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

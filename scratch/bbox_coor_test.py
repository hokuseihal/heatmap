from head import *
'''
Predint Bounding BOX coordinates.Box is given as 0/1 map. Accuracy is not so good...
'''
numepoch = 500
batchsize = 32
data = torch.zeros((1, 128, 128), dtype=bool)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn1 = nn.Conv2d(1, 4, 65)
        self.cnn2 = nn.Conv2d(4, 8, 33)
        self.fc1=nn.Linear(8192,4096)
        self.fc2=nn.Linear(4096,4)

    def forward(self, x):
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x=x.view(batchsize,-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = x.view(-1, 4)
        return x


net = Net()
optimizer = torch.optim.Adam(net.parameters())


def getitem(n):
    retx = []
    rety = []
    for _ in range(n):
        randx = np.sort(np.random.randint(0, 128, size=2))
        randy = np.sort(np.random.randint(0, 128, size=2))
        data[:, randx[0]:randy[0], randx[1]:randy[1]] = True
        retx.append(data)
        rety.append(torch.cat([torch.from_numpy(randx), torch.from_numpy(randy)]))
    return torch.stack(retx).type(torch.float32), torch.stack(rety).type(torch.float32) / 128

writer = SummaryWriter()

for e in range(numepoch):
    optimizer.zero_grad()
    input, target = getitem(batchsize)
    output = net(input)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    print(f'{e}Train:{loss.item()}')
    writer.add_scalar('Loss:Train', loss.item(), e)

    with torch.no_grad():
        optimizer.zero_grad()
        input, target = getitem(batchsize)
        output = net(input)
        loss = F.mse_loss(output, target)
        print(f'{e}Test:{loss.item()}')
        writer.add_scalar('Loss:Test', loss.item(), e)

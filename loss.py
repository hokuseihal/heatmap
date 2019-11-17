import torch
import torch.nn as nn
class mask_mse_loss(nn.Module):
    def __init__(self):
        super(mask_mse_loss,self).__init__()

    def forward(self,input,target):
        assert input.shape==target.shape
        #TODO 報酬は？
        #正解０、マスク１、間違い２
        return 2*target if x==0



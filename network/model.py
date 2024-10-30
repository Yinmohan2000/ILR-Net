import torch
import torch.nn as  nn
from network.retinex import DecomNet,Adjust_naive
from network.enhance import LRelight
from network.Attention import ECA_block
class demo(nn.Module):
    def __init__(self):
        super(demo,self).__init__()
        self.demo = DecomNet()
        self.adjust = Adjust_naive()

    def forward(self,x):
        c,a,b = self.demo(x)
        out = self.adjust(a)
        return out
class ILR(nn.Module):
    def __init__(self):
        super(ILR,self).__init__()
        self.eca = ECA_block(16)
        self.decom_net = demo()
        self.Lrelight_net = LRelight()
        self.conv = nn.Conv2d(32,3,3,1,1)

    def forward(self,x):
        x = x.permute(0,3,1,2)
        outR = self.decom_net(x)
        outL = self.Lrelight_net(x)
        out1 = torch.cat([outR,outL],1)
        out = self.eca(out1)
        outall = self.conv(out)
        return outall.permute(0,2,3,1)

if __name__ == '__main__':
    net = ILR()
    data_in = torch.rand(1, 600, 400, 3)
    out = net(data_in)
    print('--------------------------------')
    print(out.size())

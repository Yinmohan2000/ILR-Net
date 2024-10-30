import torch
import torch.nn as nn
from network.Attention import se_block,ca_block,Residual
class DecomNet(nn.Module):
    def __init__(self, layer_num=2, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.layer_num = layer_num
        self.conv0 = nn.Conv2d(3, channel, kernel_size*3, padding=4)
        self.ca    = ca_block(channel,channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1,padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1,padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1,padding_mode='replicate'),
            nn.ReLU(),
            se_block(channel)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1,padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1,padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1,padding_mode='replicate'),
            nn.ReLU(),
            se_block(channel)
        )
        self.res  = Residual(channel,channel)
        self.conv1 = nn.Conv2d(channel, 4, kernel_size, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv0(x)
        out2 = self.ca(out1)
        out3 =out1+out2
        out4 = self.conv2(out3)
        out5 = out3+out4
        out5 = self.conv3(out5)
        out6 = self.res(out5)
        out = self.conv1(out6)
        out = self.sig(out)
        r_part = out[:, 0:3, :, :]
        l_part = out[:, 3:4, :, :]
        return out, r_part,l_part
class Adjust_naive(nn.Module):
    def __init__(self,channel = 64,kernel_size = 3):
        super(Adjust_naive,self).__init__()
        self.conv1 = nn.Conv2d(3, channel, kernel_size, 1, 1, padding_mode='replicate')
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, 1, 1,padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size, 1, 1,padding_mode='replicate'),
            nn.ReLU(inplace=True),
            se_block(channel)
        )
        self.conv3 = nn.Conv2d(channel, channel, kernel_size, 1, 1,padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel, 16, kernel_size, 1, 1,padding_mode='replicate')
        self.se    = se_block(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.se(x)
        x_1 = self.conv2(x)
        add1 = x_1 + x
        x_2 = self.conv2(add1)
        add2 = x_2 + add1
        x_3 = self.conv2(add2)
        add3 = x_3 + add2
        out = self.relu(self.conv3(add3))
        out = self.conv4(out)
        return out
    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt

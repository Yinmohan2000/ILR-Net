import torch.nn as nn
import torch
from network.Attention import se_block
from network.unet import RelightNet
class PZConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PZConv, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=4, padding=4)
        self.conv8 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=8, padding=8)
        self.conv = nn.Conv2d(out_channels * 4, 16, kernel_size=3, dilation=1, padding=1)

    def forward(self, x):
        out0 = self.conv0(x)
        out1 = self.conv1(out0)
        out2 = self.conv2(out0)
        out3 = self.conv4(out0)
        out4 = self.conv8(out0)
        out = torch.cat((out1, out2, out3, out4), 1)
        out_all = self.conv(out)
        return out_all
class Enhancement(nn.Module):
    def __init__(self, channel=64):
        super(Enhancement, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(16, channel, 3, 1, 1, padding_mode='replicate')
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, padding_mode='replicate'),
            nn.BatchNorm2d(channel),
        )
        self.conv2 = nn.Conv2d(channel, 16, 3, 1, 1, padding_mode='replicate')
        self.sig = nn.Sigmoid()

    def forward(self, x):
        conv0 = self.relu(self.conv0(x))
        conv1 = self.relu(self.conv1(conv0))
        out1 = conv1 + conv0
        conv2 = self.conv2(out1)
        out = self.sig(conv2)
        out_all = out + x
        return out_all
class Correct(nn.Module):
    def __init__(self):
        super(Correct, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1, padding_mode='replicate')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, padding_mode='replicate')
        self.relu2 = nn.ReLU(inplace=True)
        self.se_layer = se_block(channel=64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='replicate'),
            nn.Conv2d(128, 128, 3, 1, 1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, padding_mode='replicate'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Conv2d(128, 64, 3, 1, 1, padding_mode='replicate')
        self.conv9 = nn.Conv2d(64, 16, 3, 1, 1, padding_mode='replicate')
        self.sig = nn.Sigmoid()

    def forward(self, r, l):
        r_fs = self.relu1(self.conv1(r))
        l_fs = self.relu2(self.conv2(l))
        inf = torch.cat([r_fs, l_fs], dim=1)
        se_inf = self.se_layer(inf)
        x1 = self.conv3(se_inf)
        x2 = self.conv(x1)
        x1_2 = x1 + x2
        x3 = self.conv(x1_2)
        x3_1_1 = x3 + x1_2
        x4 = self.conv(x3_1_1)
        x4_3_1_1 = x4 + x3_1_1
        x5 = self.conv8(x4_3_1_1)
        x6 = self.conv9(x5)
        n = self.sig(x6)
        r_restore = r + n
        return r_restore
class LRelight(nn.Module):
    def __init__(self):
        super(LRelight, self).__init__()
        self.dconv = PZConv(3, 64)
        self.enhance = Enhancement()
        self.correct = Correct()
        self.relight = RelightNet()
        self.conv = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, x):
        outD = self.dconv(x)
        outA = self.relight(outD)
        outE = self.enhance(outD)
        out1 = outD * outE
        outC = self.correct(out1, outA)
        add1 = outD + outC
        outE_1 = self.enhance(add1)
        out2 = outD * outE_1
        outC_1 = self.correct(out2, outA)
        add2 = outD + outC_1
        outE_2 = self.enhance(add2)
        out3 = outD * outE_2
        outC_2 = self.correct(out3, outA)
        return outC_2

# if __name__ == '__main__':
#     data_in = torch.rand(1, 600, 400, 3)
#     ut_S, a, b = LRelight(data_in)
#     print('--------------------------------')
#     print(ut_S.size())
#     print(a.size())
#     print(b.size())

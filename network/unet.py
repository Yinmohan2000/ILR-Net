import torch
import torch.nn as nn
import torch.nn.functional as F
from network.Attention import Residual,GFA
class RelightNet(nn.Module):
    def __init__(self, channel=32, kernel_size=3):
        super(RelightNet, self).__init__()
        self.conv0 = nn.Conv2d(16, channel, kernel_size, padding=1,padding_mode='replicate')
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel*2, kernel_size, stride=2, padding=1,padding_mode='replicate'),
            nn.Conv2d(channel*2, channel*2, kernel_size, stride=1, padding=1,padding_mode='replicate'),
            Residual(channel*2,channel*2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel*2, channel*4, kernel_size, stride=2, padding=1,padding_mode='replicate'),
            nn.Conv2d(channel*4, channel * 4, kernel_size, stride=1, padding=1,padding_mode='replicate'),
            Residual(channel*4, channel*4),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel*4, channel*8, kernel_size, stride=2, padding=1,padding_mode='replicate'),
            nn.Conv2d(channel*8, channel * 8, kernel_size, stride=1, padding=1,padding_mode='replicate'),
            Residual(channel*8, channel*8),
        )
        self.gfa = GFA(channel*8)
        self.deconv1 = nn.Sequential(
            Residual(channel*8, channel*8),
            nn.Conv2d(channel*8, channel*4, kernel_size, padding=1,padding_mode='replicate'),
            nn.Conv2d(channel*4, channel*4, kernel_size, padding=1,padding_mode='replicate'),
        )
        self.deconv2 = nn.Sequential(
            Residual(channel*4, channel*4),
            nn.Conv2d(channel*4, channel*2, kernel_size, padding=1,padding_mode='replicate'),
            nn.Conv2d(channel*2, channel*2, kernel_size, padding=1,padding_mode='replicate'),
        )
        self.deconv3 = nn.Sequential(
            Residual(channel*2, channel*2),
            nn.Conv2d(channel*2, channel, kernel_size, padding=1,padding_mode='replicate'),
            nn.Conv2d(channel, channel, kernel_size, padding=1,padding_mode='replicate'),
        )
        self.feature_fusion = nn.Conv2d(channel*7, channel, 1)
        self.output = nn.Conv2d(channel, 16, kernel_size, padding=1,padding_mode='replicate')

    def forward(self, x):
        conv0 = self.lrelu(self.conv0(x))
        conv1 = self.lrelu(self.conv1(conv0))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.lrelu(self.conv3(conv2))
        conv3 = self.gfa(conv3)
        up1 = F.interpolate(conv3, scale_factor=2)
        deconv1 = self.lrelu(self.deconv1(up1)) + conv2
        up2 = F.interpolate(deconv1, scale_factor=2)
        deconv2 = self.lrelu(self.deconv2(up2)) + conv1
        up3 = F.interpolate(deconv2, scale_factor=2)
        deconv3 = self.lrelu(self.deconv3(up3)) + conv0
        deconv1_resize = F.interpolate(deconv1, scale_factor=4)
        deconv2_resize = F.interpolate(deconv2, scale_factor=2)
        out = torch.cat((deconv1_resize, deconv2_resize, deconv3), dim=1)
        out = self.feature_fusion(out)
        out = self.output(out)
        return out
    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt

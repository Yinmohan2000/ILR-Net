import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from network.model import ILR
from dataset import TheDataset
from network.Loss import CharbonnierLoss,ReflectLoss,PerceptualLoss,ssim
import tqdm

parser = argparse.ArgumentParser(description='ILR-Net args setting')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default='0', help='GPU idx')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='number of total epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--workers', dest='workers', type=int, default=0, help='num workers of dataloader')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--save_interval', dest='save_interval', type=int, default=20, help='save model every # epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test/low', help='directory for testing inputs')
parser.add_argument('--decom', dest='decom', default=0,
                    help='decom flag, 0 for enhanced results only and 1 for decomposition results')

args = parser.parse_args()
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)
LRelight_net = ILR()
if args.use_gpu:
    LRelight_net = LRelight_net.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True
lr = args.start_lr * np.ones([args.epoch])
lr[20:] = lr[0] / 10.0
LRelight_optim = torch.optim.Adam(LRelight_net.parameters(), lr=args.start_lr)
train_set = TheDataset()

ssim = ssim(11,0.01)
Char_Loss = CharbonnierLoss()
Reflect_Loss = ReflectLoss()
Perceptual_Loss = PerceptualLoss().cuda()
def train():
    LRelight_net.train()
    for epoch in range(args.epoch):
        times_per_epoch, sum_loss = 0, 0.
        dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=0, pin_memory=True)
        LRelight_optim.param_groups[0]['lr'] = lr[epoch]
        for data in tqdm.tqdm(dataloader):
            times_per_epoch += 1
            low_im, high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()
            LRelight_optim.zero_grad()
            delta = LRelight_net(low_im)
            loss_ssim = ssim(delta,high_im)
            loss_pre = Perceptual_Loss(delta,high_im)
            loss_cc = Char_Loss(delta,high_im)
            loss_reflect = Reflect_Loss(low_im,delta)
            loss_total = loss_reflect + loss_pre + loss_ssim + loss_cc
            loss_total.backward()
            LRelight_optim.step()
            sum_loss += loss_total
        print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss / times_per_epoch))
        if (epoch+1) % args.save_interval == 0:
            torch.save(LRelight_net.state_dict(), args.ckpt_dir + '/TILR_net_' + str(epoch) + '.pth')
    torch.save(LRelight_net.state_dict(), args.ckpt_dir + '/TILR_net_final.pth')

if __name__ == '__main__':
    train()

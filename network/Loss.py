import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import pytorch_ssim
def gradient(input_tensor, direction):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    h, w = input_tensor.size()[2], input_tensor.size()[3]
    smooth_kernel_x = torch.reshape(torch.Tensor([[0., 0.], [-1., 1.]]).cuda(), (1, 1, 2, 2))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    assert direction in ['x', 'y']
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y
    out = F.conv2d(input_tensor, kernel, padding=(1, 1))
    out = torch.abs(out[:, :, 0:h, 0:w])
    return out.permute(0, 2, 3, 1)
def ave_gradient(input_tensor, direction):
    return (F.avg_pool2d(gradient(input_tensor, direction).permute(0, 3, 1, 2), 3, stride=1, padding=1))\
        .permute(0, 2, 3, 1)
def smooth(input_l, input_r):
    rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140]).cuda()
    input_r = torch.tensordot(input_r, rgb_weights, dims=([-1], [-1]))
    input_r = torch.unsqueeze(input_r, -1)
    return torch.mean(
        gradient(input_l, 'x') * torch.exp(-10 * ave_gradient(input_r, 'x')) +
        gradient(input_l, 'y') * torch.exp(-10 * ave_gradient(input_r, 'y'))
    )
def smooth_enhance(delta, input):
    rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140]).cuda()
    input = torch.tensordot(input, rgb_weights, dims=([-1], [-1]))
    input = torch.unsqueeze(input, -1)
    return torch.mean(
        gradient(delta, 'x') * torch.exp(-10 * ave_gradient(input, 'x')) +
        gradient(delta, 'y') * torch.exp(-10 * ave_gradient(input, 'y'))
    )
class enhance_smooth(nn.Module):
    def __init__(self):
        super(enhance_smooth,self).__init__()
    def forward(self,input,delta):
        input_low_gray = torch.mean(delta, dim=1, keepdim=True)

        input_low_gray_3 = torch.cat([input_low_gray, input_low_gray, input_low_gray], dim=1)
        ismooth_loss = smooth_enhance(input_low_gray_3, input)
        return ismooth_loss
class DecomLoss(nn.Module):
    def __init__(self):
        super(DecomLoss, self).__init__()

    def forward(self, r_low, l_low, r_high, l_high, input_low, input_high):
        l_low_3 = torch.cat((l_low, l_low, l_low), -1)
        l_high_3 = torch.cat((l_high, l_high, l_high), -1)
        recon_loss_low = torch.mean(torch.abs(r_low * l_low_3 - input_low))
        recon_loss_high = torch.mean(torch.abs(r_high * l_high_3 - input_high))
        recon_loss_mutal_low = torch.mean(torch.abs(r_high * l_low_3 - input_low))
        recon_loss_mutal_high = torch.mean(torch.abs(r_low * l_high_3 - input_high))
        equal_r_loss = torch.mean(torch.abs(r_low - r_high))
        ismooth_loss_low = smooth(l_low, r_low)
        ismooth_loss_high = smooth(l_high, r_high)
        return \
            recon_loss_low + recon_loss_high +\
            0.001*recon_loss_mutal_low + 0.001*recon_loss_mutal_high + \
            0.1*ismooth_loss_low + 0.1*ismooth_loss_high + \
            0.01*equal_r_loss
class CharbonnierLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, input_high):
        diff = x - input_high
        loss = torch.sqrt(diff.pow(2) + self.epsilon)
        return torch.mean(loss)
class ReflectLoss(nn.Module):
    def __init__(self):
        super(ReflectLoss, self).__init__()

    def forward(self, input_image, output_image):
        # Assuming input and output images are tensors of shape (batch_size, channels, height, width)
        # Extracting the reflectance component of the images (e.g., using luminance)
        reflectance_input = self.luminance_extraction(input_image)
        reflectance_output = self.luminance_extraction(output_image)
        # Calculating the squared Euclidean distance between reflectance components
        loss = torch.mean(torch.pow(reflectance_input - reflectance_output, 2))
        return loss
    def luminance_extraction(self, image):
        # For simplicity, assuming luminance extraction as taking the mean across color channels
        # Modify this according to the specific method for extracting reflectance in your context
        return torch.mean(image, dim=1)
def vgg16_loss(feature_module,loss_func,y,y_):
    out=feature_module(y)
    out_=feature_module(y_)
    loss=loss_func(out,out_)
    return loss
def get_feature_module(layer_index,device=None):
    vgg = vgg16(pretrained=True, progress=True).features
    vgg.eval()
    for parm in vgg.parameters():
        parm.requires_grad = False
    feature_module = vgg[0:layer_index + 1]
    feature_module.to(device)
    return feature_module
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        loss_func = nn.MSELoss()
        layer_indexs = [3,14]
        device = torch.device("cuda")
        self.creation=loss_func
        self.layer_indexs=layer_indexs
        self.device=device

    def forward(self,y,y_):
        y = y.permute(0, 3, 1, 2)
        y_ = y_.permute(0, 3, 1, 2)
        loss=0
        for index in self.layer_indexs:
            feature_module=get_feature_module(index,self.device)
            loss+=vgg16_loss(feature_module,self.creation,y,y_)
        return loss
class ssim(nn.Module):
    def __init__(self,window_size = 11, ssim_weight = 1):
        super().__init__()
        self.window_size = window_size
        self.ssim_weight = ssim_weight
    def forward(self,delta,input_high):
        # delta = delta.permute(0, 3, 1, 2)
        # input_high = input_high.permute(0, 3, 1, 2)
        input_high_gray = torch.mean(input_high, dim=1, keepdim=True)
        input_low_gray = torch.mean(delta, dim=1, keepdim=True)
        input_low_gray_3 = torch.cat([input_low_gray, input_low_gray, input_low_gray], dim=1)
        input_high_gray_3 = torch.cat([input_high_gray, input_high_gray, input_high_gray], dim=1)
        ssim_weight = self.ssim_weight
        ssim_loss = 1 - pytorch_ssim.SSIM(window_size=self.window_size)(input_low_gray_3, input_high_gray_3)
        return ssim_loss * ssim_weight

# if __name__ == '__main__':
#     tensor = torch.rand(1, 300, 400, 1)
#     out_data = ssim(tensor, torch.rand(1, 300, 400, 3))
#     print(out_data)

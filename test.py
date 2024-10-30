from network.model import ILR
import torch
from PIL import Image
import numpy as np
import cv2

with torch.no_grad():
    input = Image.open(r"D:\PycharmProjects\YCrBr\data\eval\eval15\low\LOL\1.png")
    input = np.asarray(input)
    input_c = input.copy()
    input = np.array(input, dtype="float32")/255.0
    input = torch.from_numpy(input).view(1, 400,600, 3).cuda()  # 图像大小统一为600 * 400
    print(input.shape)
    enhance_net = ILR().cuda()
    enhance_net.load_state_dict(torch.load(r"D:\PycharmProjects\yinmohan\checkpoint\ILR_net_final.pth"))
    #enhance_net.load_state_dict(torch.load(r"D:\PycharmProjects\yinmohan\checkpoint\ILR_net_final_1.pth"))
    out_S = enhance_net(input)
    out_S = np.squeeze(out_S, axis=0)
    out_S = out_S.cpu().detach().numpy()
    out = out_S
    im = np.clip(out * 255.0, 0, 255.0).astype('uint8')
    im = im[:,:,[2, 1, 0]]
    cv2.imwrite(r"D:\PycharmProjects\yinmohan\result\000.png", im)



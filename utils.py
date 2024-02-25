"""
Author  : Xu fuyong
Time    : created by 2019/7/16 20:14

"""
import torch
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from math import exp




def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def calc_ssim(image1, image2, window_size=11, size_average=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 调整图像维度和类型
    image1 = image1.float()
    image2 = image2.float()

    # 转换权重张量类型
    weight = torch.ones(image1.shape[1], 1, window_size, window_size).to(image1.device) / (window_size ** 2)

    # 计算均值
    mu1 = F.conv2d(image1, weight=weight, padding=window_size//2, groups=image1.shape[1])
    mu2 = F.conv2d(image2, weight=weight, padding=window_size//2, groups=image2.shape[1])

    # 计算方差
    sigma1 = F.conv2d(image1 ** 2, weight=weight, padding=window_size//2, groups=image1.shape[1]) - mu1 ** 2
    sigma2 = F.conv2d(image2 ** 2, weight=weight, padding=window_size//2, groups=image2.shape[1]) - mu2 ** 2
    sigma12 = F.conv2d(image1 * image2, weight=weight, padding=window_size//2, groups=image1.shape[1]) - mu1 * mu2

    # 计算 SSIM 分子部分
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    # 计算 SSIM 分母部分
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

    # 计算 SSIM
    ssim_map = numerator / denominator
    if size_average:
        ssim_score = ssim_map.mean()
    else:
        ssim_score = ssim_map.mean(1).mean(1).mean(1)

    return ssim_score.item()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window




class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
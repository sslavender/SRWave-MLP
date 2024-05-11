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


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.stack([y, cb, cr],dim= 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 20. * torch.log10(1. / torch.sqrt(torch.mean((img1 - img2) ** 2)))

def calc_ssim(image1, image2, window_size=11, size_average=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    #Adjusting image dimensions and types
    image1 = image1.float()
    image2 = image2.float()

    #Convert weight tensor types
    weight = torch.ones(image1.shape[1], 1, window_size, window_size).to(image1.device) / (window_size ** 2)

     # Calculate Mean
    mu1 = F.conv2d(image1, weight=weight, padding=window_size//2, groups=image1.shape[1])
    mu2 = F.conv2d(image2, weight=weight, padding=window_size//2, groups=image2.shape[1])

    #Calculate variance
    sigma1 = F.conv2d(image1 ** 2, weight=weight, padding=window_size//2, groups=image1.shape[1]) - mu1 ** 2
    sigma2 = F.conv2d(image2 ** 2, weight=weight, padding=window_size//2, groups=image2.shape[1]) - mu2 ** 2
    sigma12 = F.conv2d(image1 * image2, weight=weight, padding=window_size//2, groups=image1.shape[1]) - mu1 * mu2

    #Calculate the molecular part of SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    #Calculate the denominator part of SSIM
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

    #Calculate SSIM
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
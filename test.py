"""
Author  : Xu fuyong
Time    : created by 2019/7/17 17:41me/dell/code/ml

"""
import argparse
  
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms.functional as F
import PIL.Image as pil_image
import glob
import os

from model10_1_1_2 import SRWaveMLP
from utils import  calc_psnr , calc_ssim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str,default='/home/dell/code/mlp/outputs/weight_2.23.pth')
    parser.add_argument('--images-dir', type=str, default='/home/dell/code/mlp/bsds100/original')
    parser.add_argument('--outputs', type=str, default='/home/dell/code/mlp/test/bsds100')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    cudnn.benchmark = True
    # device = torch.device('cuda: 1' if torch.cuda.is_available() else 'cpu')



# 加载预训练模型
    transitions = [True, True, True, True]
    layers = [3,3]
    mlp_ratios = [4,4]
        # embed_dims = [64, 128, 320, 512]
    embed_dims = [64,64]
    model = SRWaveMLP(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        mlp_ratios=mlp_ratios,mode='depthwise',scale=4).cuda()

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)


# 将模型设置为评估模式
    model.eval()
    psnr = 0.
    total = 0
    ssim = 0.

for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
# 读取输入图像并进行预处理
    image = pil_image.open(image_path).convert('RGB')
    image_width = (image.width // (args.scale**2)) * (args.scale**2)
    image_height = (image.height //  (args.scale**2)) *  (args.scale**2)

    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image_hr = F.to_tensor(image).unsqueeze(0).cuda()
    simage = image
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    z = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)

    file_name = os.path.basename(image_path)
    new_file_path = os.path.join(args.outputs, file_name)
    # image.save(new_file_path.replace('.', '_bicubic_x{}.'.format(args.scale)))


# 将输入图像转换为模型需要的格式并进行超分辨率重建
    image = np.array(image).astype(np.float32)
    simage = np.array(simage).astype(np.float32)
    simage = simage/255.
    simage = torch.from_numpy(simage).cuda()
    simage = simage.unsqueeze(0)

    
    
    z =  np.array(z).astype(np.float32)
    z = z/255.
    z = torch.from_numpy(z).cuda()
    z = z.unsqueeze(0)

    
    y = image/255.
    y = torch.from_numpy(y).cuda()
    y = y.unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)


# 计算重建图像的PSNR值
    spsnr = calc_psnr(preds,simage)

    ssim_score = calc_ssim(preds.permute(0,3,1,2),image_hr)
    psnr += spsnr
    total += 1
    ssim += ssim_score

#  # 对重建图像进行后处理并保存  
#     preds = preds.mul(255.0).cpu().numpy().squeeze(0)
#     output = np.clip(preds , 0.0, 255.0).astype(np.uint8)
#     output = pil_image.fromarray(output)
#     output.save(new_file_path.replace('.', '_mlp_x{}.'.format(spsnr)))
mpsnr = psnr / total
mssim = ssim / total
print(mpsnr) 
print(mssim) 
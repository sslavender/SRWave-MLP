
import os
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import to_2tuple

import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from gated_fusion import Gated

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, dropout_rate=0.3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        avg_pool = self.dropout(avg_pool)
        channel_attention = self.fc(avg_pool)
        channel_attention = channel_attention.unsqueeze(2).unsqueeze(3)
        channel_attention = channel_attention.expand_as(x)
        return x * channel_attention


class Up(nn.Module):
    def __init__(self, n_feat):
        super(Up, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
class UpSample(nn.Module):
    def __init__(self, scale_factor, in_channels=32):
        super().__init__()
        self.up = nn.Sequential(
           
            nn.Conv2d(in_channels, 3*(scale_factor**2), kernel_size=1),
            nn.PixelShuffle(scale_factor)
        )
        self._initialize_weight()

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0,
                                    std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.up(x)
        return x.permute(0, 2, 3, 1)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'wave_T': _cfg(crop_pct=0.9),
    'wave_S': _cfg(crop_pct=0.9),
    'wave_M': _cfg(crop_pct=0.9),
    'wave_B': _cfg(crop_pct=0.875),
}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   


class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,mode='fc'):
        super().__init__()
        
        
        self.fc_h = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias) 
        self.fc_c = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        
        self.tfc_h = nn.Conv2d(2*dim, dim, (1,7), stride=1, padding=(0,7//2), groups=dim, bias=False) 
        self.tfc_w = nn.Conv2d(2*dim, dim, (7,1), stride=1, padding=(7//2,0), groups=dim, bias=False)  
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)   
        
        if mode=='fc':
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())  
        else:
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU()) 
                    


    def forward(self, x):
     
        B, C, H, W = x.shape
        theta_h=self.theta_h_conv(x)
        theta_w=self.theta_w_conv(x)

        x_h=self.fc_h(x)
        x_w=self.fc_w(x)      
        x_h=torch.cat([x_h*torch.cos(theta_h),x_h*torch.sin(theta_h)],dim=1)
        x_w=torch.cat([x_w*torch.cos(theta_w),x_w*torch.sin(theta_w)],dim=1)

#         x_1=self.fc_h(x)
#         x_2=self.fc_w(x)
#         x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
#         x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)
        
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c,output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)           
        return x
        
class WaveBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.3, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop,mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.cattn = ChannelAttention(dim)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) 
        x = x + self.drop_path(self.mlp(self.cattn(self.norm2(x)))) 
        return x


class PatchEmbedOverlapping(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d, groups=1,use_norm=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)      
        self.norm = norm_layer(embed_dim) if use_norm==True else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0, out_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d, groups=1, use_norm=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.deproj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
        self.norm = norm_layer(out_chans) if use_norm else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = self.deproj(x)
        return x


class Connection(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size,norm_layer=nn.BatchNorm2d,use_norm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.norm = norm_layer(out_embed_dim) if use_norm==True else nn.Identity()
    def forward(self, x):
        x = self.proj(x) 
        x = self.norm(x)
        return x        


def basic_blocks(dim, index, layers, mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0.,norm_layer=nn.BatchNorm2d,mode='fc', **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(WaveBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, norm_layer=norm_layer,mode=mode))
    blocks = nn.Sequential(*blocks)
    return blocks

class SRWaveMLP(nn.Module):
    def __init__(self, layers, img_size=36, patch_size=4, in_chans=3, num_classes=1000,
        embed_dims=None, transitions=None, mlp_ratios=None, 
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.BatchNorm2d, fork_feat=False,mode='fc',ds_use_norm=True,args=None,scale = 2): 

        super(SRWaveMLP, self).__init__()
        
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbedOverlapping(patch_size=1, stride=1, padding=0, in_chans=3, embed_dim=embed_dims[0],norm_layer=norm_layer,use_norm=ds_use_norm)
        self.patch_embed1 = PatchEmbedOverlapping(patch_size=1, stride=1, padding=0, in_chans=3, embed_dim=embed_dims[0],norm_layer=norm_layer,use_norm=ds_use_norm)
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer,mode=mode)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            network.append(Connection(embed_dims[i], embed_dims[i+1], 3,norm_layer=norm_layer,use_norm=ds_use_norm))

        self.network = nn.ModuleList(network)
        network1 = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer,mode=mode)
            network1.append(stage)
            if i >= len(layers) - 1:
                break
            network1.append(Connection(embed_dims[i], embed_dims[i+1], 3,norm_layer=norm_layer,use_norm=ds_use_norm))

        self.network1 = nn.ModuleList(network1)
        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1]) 
            self.norm1 = norm_layer(embed_dims[-1])
        self.apply(self.cls_init_weights)

        self.upsample_layers = UpSample(in_channels=embed_dims[len(layers) - 1], scale_factor=scale)
        self.Expanddim = nn.Conv2d(3,embed_dims[0],1)
        self.chan =  ChannelAttention(embed_dims[len(layers) - 1])
        self.Expanddim1 = nn.Sequential(nn.Conv2d(3,embed_dims[len(layers) - 1]//2,1),
                                        ChannelAttention(embed_dims[len(layers) - 1]//2))
        self.Expanddim2 = nn.Conv2d(3,128,1)
        self.Reducedim = nn.Conv2d(128,64,1)
   
        self.gate_layers = Gated(embed_dims[len(layers) - 1])
        self.gate_layers1 = Gated(3)
        self.up = Up(embed_dims[len(layers) - 1])

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None): 
        """ mmseg or mmdet `init_weight` """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)



    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x
    def second_embeddings(self, x):
        x = self.patch_embed1(x)
        return x

    def second_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network1):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def upsample(self,x):
        return self.upsample_layers(x)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        inter_res = nn.functional.interpolate(
            x,
            scale_factor= 4,
            mode='bicubic',
            align_corners=False)
        
        inter_res1 = nn.functional.interpolate(
            x,
            scale_factor= 0.5,
            mode='bilinear',
            align_corners=False)
        inter_res2 = nn.functional.interpolate(
            inter_res1,
            scale_factor= 2,
            mode='bilinear',
            align_corners=False)

        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)

        inter_res1 = self.second_embeddings(inter_res1)
        inter_res1 = self.second_tokens(inter_res1)
        inter_res1 = self.norm1(inter_res1)
        inter_res1 = self.up(inter_res1)
 

        inter_res2 = self.Expanddim1(inter_res2)
        inter_res1 = torch.cat([inter_res2, inter_res1], 1)

        out = self.gate_layers(x,inter_res1)
        out = self.chan(out)
        cls_out = self.upsample(out)
        cls_out = cls_out.permute(0,3,1,2)
        cls_out = self.gate_layers1(inter_res,cls_out).permute(0,2,3,1)

        return cls_out

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math
import numpy as np


class Intervolution(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1, dilation=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5

        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        # self.attn = nn.Linear(dim, kernel_size**4 * num_heads)
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=kernel_size**4 * num_heads, kernel_size=1),
            nn.BatchNorm2d(kernel_size**4 * num_heads),
            nn.ReLU()
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=dilation*(kernel_size-1)//2, stride=stride, dilation=dilation)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, q, s):
        B, H, W, C = q.shape

        # v = self.v(q).permute(0, 3, 1, 2)  # B, C, H, W
        v = self.v(q.permute(0, 3, 1, 2))  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        # attn = self.pool(s.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # attn = self.attn(attn).reshape(
        #     B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
        #     self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = self.pool(s.permute(0, 3, 1, 2))
        attn = self.attn(attn).permute(0, 2, 3, 1).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


class Interlooker(nn.Module):
    def __init__(self, dim, out_dim, kernel_size, padding, stride=1, dilation=1,
                 num_heads=1, mlp_ratio=3., attn_drop=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, qkv_bias=False,
                 qk_scale=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Intervolution(dim, num_heads, kernel_size=kernel_size,
                                  padding=padding, stride=stride, dilation=dilation,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=attn_drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       out_features=out_dim)

    def forward(self, q, s):
        x = q + self.attn(self.norm1(q), self.norm1(s))
        x = self.mlp(self.norm2(x))
        return x
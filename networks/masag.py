import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import functools
import math
import timm
from timm.models.layers import DropPath, to_2tuple
import einops
from fvcore.nn import FlopCountAnalysis


def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums

class GlobalExtraction(nn.Module):
  def __init__(self,dim = None):
    super().__init__()
    self.avgpool = self.globalavgchannelpool
    self.maxpool = self.globalmaxchannelpool
    self.proj = nn.Sequential(
        nn.Conv2d(2, 1, 1,1),
        nn.BatchNorm2d(1)
    )
  def globalavgchannelpool(self, x):
    x = x.mean(1, keepdim = True)
    return x

  def globalmaxchannelpool(self, x):
    x = x.max(dim = 1, keepdim=True)[0]
    return x

  def forward(self, x):
    x_ = x.clone()
    x = self.avgpool(x)
    x2 = self.maxpool(x_)

    cat = torch.cat((x,x2), dim = 1)

    proj = self.proj(cat)
    return proj

class ContextExtraction(nn.Module):
  def __init__(self, dim, reduction = None):
    super().__init__()
    self.reduction = 1 if reduction == None else 2

    self.dconv = self.DepthWiseConv2dx2(dim)
    self.proj = self.Proj(dim)

  def DepthWiseConv2dx2(self, dim):
    dconv = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 1,
              groups = dim),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 2,
              dilation = 2),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True)
    )
    return dconv

  def Proj(self, dim):
    proj = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim //self.reduction,
              kernel_size = 1
              ),
        nn.BatchNorm2d(num_features = dim//self.reduction)
    )
    return proj
  def forward(self,x):
    x = self.dconv(x)
    x = self.proj(x)
    return x

class MultiscaleFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.local= ContextExtraction(dim)
    self.global_ = GlobalExtraction()
    self.bn = nn.BatchNorm2d(num_features=dim)

  def forward(self, x, g,):
    x = self.local(x)
    g = self.global_(g)

    fuse = self.bn(x + g)
    return fuse


class MultiScaleGatedAttn(nn.Module):
    # Version 1
  def __init__(self, dim):
    super().__init__()
    self.multi = MultiscaleFusion(dim)
    self.selection = nn.Conv2d(dim, 2,1)
    self.proj = nn.Conv2d(dim, dim,1)
    self.bn = nn.BatchNorm2d(dim)
    self.bn_2 = nn.BatchNorm2d(dim)
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=dim, out_channels=dim,
                  kernel_size=1, stride=1))

  def forward(self,x,g):
    x_ = x.clone()
    g_ = g.clone()

    #stacked = torch.stack((x_, g_), dim = 1) # B, 2, C, H, W

    multi = self.multi(x, g) # B, C, H, W

    ### Option 2 ###
    multi = self.selection(multi) # B, num_path, H, W

    attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
    #attention_weights = torch.sigmoid(multi)
    A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

    x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
    g_att = B.expand_as(g_) * g_

    x_att = x_att + x_
    g_att = g_att + g_
    ## Bidirectional Interaction

    x_sig = torch.sigmoid(x_att)
    g_att_2 = x_sig * g_att


    g_sig = torch.sigmoid(g_att)
    x_att_2 = g_sig * x_att

    interaction = x_att_2 * g_att_2

    projected = torch.sigmoid(self.bn(self.proj(interaction)))

    weighted = projected * x_

    y = self.conv_block(weighted)

    #y = self.bn_2(weighted + y)
    y = self.bn_2(y)
    return y

if __name__ == "__main__":
    xi = torch.randn(1, 192, 28, 28).cuda()
    #xi_1 = torch.randn(1, 384, 14, 14)
    g = torch.randn(1, 192, 28, 28).cuda()
    #ff = ContextBridge(dim=192)

    attn = MultiScaleGatedAttn(dim = xi.shape[1]).cuda()

    print(attn(xi, g).shape)

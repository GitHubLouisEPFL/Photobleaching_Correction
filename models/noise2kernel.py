import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single
from torch.nn.modules.conv import _ConvNd

class DonutConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True, padding_mode='zeros'):
        """Module for orthonormal convolution layer"""
        
        padding = _pair(kernel_size // 2)
        kernel_size_ = _pair(kernel_size)
        stride = _pair(1)
        dilation = _pair(1)
        transposed = False
        output_padding = _pair(0)
        groups = 1
        bias = bias
        padding_mode = padding_mode
        super().__init__(in_channels, out_channels, kernel_size_, stride, padding, dilation, transposed, \
                         output_padding, groups, bias, padding_mode)
        if self.padding_mode == 'zeros': self.padding_mode = 'constant'
        self.mask = torch.ones(self.weight.shape)
        self.mask[:,:,kernel_size//2,kernel_size//2] = 0.

    def forward(self, x):

        new_weight = self.weight * self.mask

        return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, self.padding_mode), \
                new_weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
    
    def _apply(self, fn):
        super(DonutConv2d, self)._apply(fn)
        self.mask = fn(self.mask)
        return self

class Noise2KernelModel(nn.Module):
    """simple architecture for a denoiser"""
    def __init__(self, depth=8, nb_channels=32, bias=True, p_dropout=0.0):
        
        super().__init__()

        self.first_layer = nn.ModuleList()
        self.layer_dilate2 = nn.ModuleList()
        self.layer_dilate3 = nn.ModuleList()

        self.first_layer.append(DonutConv2d(1, nb_channels, 3, padding=1, bias=bias))
        self.first_layer.append(nn.ReLU())
        self.first_layer.append(nn.Dropout(p=p_dropout))
        for i in range(depth-2):
            self.layer_dilate2.append(nn.Conv2d(nb_channels, nb_channels, 3, padding=2, bias=bias, dilation=2))
            self.layer_dilate2.append(nn.ReLU())
            self.layer_dilate2.append(nn.Dropout(p=p_dropout))
            self.layer_dilate3.append(nn.Conv2d(nb_channels, nb_channels, 3, padding=3, bias=bias, dilation=3))
            self.layer_dilate3.append(nn.ReLU())
            self.layer_dilate3.append(nn.Dropout(p=p_dropout))
        self.final_layer = nn.Conv2d(2*nb_channels, 1, 1, bias=bias)

        self.first_layer = nn.Sequential(*self.first_layer)
        self.layer_dilate2 = nn.Sequential(*self.layer_dilate2)
        self.layer_dilate3 = nn.Sequential(*self.layer_dilate3)

    def forward(self, x):
        """ """
        x = self.first_layer(x)
        x = torch.cat((self.layer_dilate2(x), self.layer_dilate3(x)), dim=1)
        x = self.final_layer(x)
        return x
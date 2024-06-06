import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _triple, _pair
from torch.nn.modules.conv import _ConvNd


class DonutConv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True, padding_mode='zeros'):
        """Module for orthonormal convolution layer"""
        
        padding = _triple(kernel_size // 2)  
        kernel_size_ = _triple(kernel_size)  
        stride = _triple(1)  
        dilation = _triple(1)  
        transposed = False
        output_padding = _triple(0)  
        groups = 1
        bias = bias
        padding_mode = padding_mode
        super().__init__(in_channels, out_channels, kernel_size_, stride, padding, dilation, transposed, \
                        output_padding, groups, bias, padding_mode)
        if self.padding_mode == 'zeros': self.padding_mode = 'constant'
        self.mask = torch.ones(self.weight.shape)
        self.mask[:,:,kernel_size//2,kernel_size//2,kernel_size//2] = 0.

    def forward(self, x):

        new_weight = self.weight * self.mask

        return F.conv3d(F.pad(x, self._reversed_padding_repeated_twice, self.padding_mode), \
                new_weight, self.bias, self.stride, _triple(0), self.dilation, self.groups)
    
    def _apply(self, fn):
        super(DonutConv3d, self)._apply(fn)
        self.mask = fn(self.mask)
        return self


class Noise2KernelModel3D(nn.Module):
    """simple architecture for a 3D denoiser"""
    def __init__(self, depth=8, nb_channels=32, bias=True, p_dropout=0.0):
        
        super().__init__()

        self.first_layer = nn.ModuleList()
        self.layer_dilate2 = nn.ModuleList()
        self.layer_dilate3 = nn.ModuleList()

        self.first_layer.append(DonutConv3d(1, nb_channels, 3, padding=1, bias=bias))
        self.first_layer.append(nn.ReLU())
        self.first_layer.append(nn.Dropout(p=p_dropout))
        for i in range(depth-2):
            self.layer_dilate2.append(nn.Conv3d(nb_channels, nb_channels, 3, padding=2, bias=bias, dilation=2))
            self.layer_dilate2.append(nn.ReLU())
            self.layer_dilate2.append(nn.Dropout(p=p_dropout))
            self.layer_dilate3.append(nn.Conv3d(nb_channels, nb_channels, 3, padding=3, bias=bias, dilation=3))
            self.layer_dilate3.append(nn.ReLU())
            self.layer_dilate3.append(nn.Dropout(p=p_dropout))
        self.final_layer = nn.Conv3d(2*nb_channels, 1, 1, bias=bias)

        self.first_layer = nn.Sequential(*self.first_layer)
        self.layer_dilate2 = nn.Sequential(*self.layer_dilate2)
        self.layer_dilate3 = nn.Sequential(*self.layer_dilate3)

    def forward(self, x):
        """ """
        x = self.first_layer(x)
        x = torch.cat((self.layer_dilate2(x), self.layer_dilate3(x)), dim=1)
        x = self.final_layer(x)
        return x


















# class DonutConv3d(_ConvNd):
#     def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True, padding_mode='zeros'):
        
#         padding = _triple(kernel_size // 2)
#         kernel_size_ = _triple(kernel_size)
#         stride = _triple(1)
#         dilation = _triple(1)
#         transposed = False
#         output_padding = _triple(0)
#         groups = 1
#         bias = bias
#         padding_mode = padding_mode

#         super().__init__(in_channels, out_channels, kernel_size_, stride, padding, dilation, transposed, \
#                          output_padding, groups, bias, padding_mode)
#         if self.padding_mode == 'zeros':
#             self.padding_mode = 'constant'
#         self.mask = torch.ones(self.weight.shape)
#         self.mask[:,:,kernel_size//2,kernel_size//2,kernel_size//2] = 0.

#     def forward(self, x):
#         new_weight = self.weight * self.mask
#         return F.conv3d(F.pad(x, self._reversed_padding_repeated_twice, self.padding_mode), \
#                 new_weight, self.bias, self.stride, _triple(0), self.dilation, self.groups)

#     def _apply(self, fn):
#         super(DonutConv3d, self)._apply(fn)
#         self.mask = fn(self.mask)
#         return self

# class Noise2KernelModel3D(nn.Module):
#     def __init__(self, depth=8, nb_channels=32, bias=True, p_dropout=0.0):
#         super().__init__()

#         self.first_layer = nn.ModuleList()
#         self.layer_dilate2 = nn.ModuleList()
#         self.layer_dilate3 = nn.ModuleList()

#         self.first_layer.append(DonutConv3d(1, nb_channels, 3, padding=1, bias=bias))
#         self.first_layer.append(nn.ReLU())
#         self.first_layer.append(nn.Dropout(p=p_dropout))
#         for i in range(depth-2):
#             self.layer_dilate2.append(nn.Conv3d(nb_channels, nb_channels, 3, padding=1, bias=bias, dilation=2))
#             self.layer_dilate2.append(nn.ReLU())
#             self.layer_dilate2.append(nn.Dropout(p=p_dropout))
#             self.layer_dilate3.append(nn.Conv3d(nb_channels, nb_channels, 3, padding=1, bias=bias, dilation=3))
#             self.layer_dilate3.append(nn.ReLU())
#             self.layer_dilate3.append(nn.Dropout(p=p_dropout))
#         self.final_layer = nn.Conv3d(2*nb_channels, 1, 1, bias=bias)

#         self.first_layer = nn.Sequential(*self.first_layer)
#         self.layer_dilate2 = nn.Sequential(*self.layer_dilate2)
#         self.layer_dilate3 = nn.Sequential(*self.layer_dilate3)

#     def forward(self, x):
#         x = self.first_layer(x)
#         x = torch.cat((self.layer_dilate2(x), self.layer_dilate3(x)), dim=1)
#         x = self.final_layer(x)
#         return x

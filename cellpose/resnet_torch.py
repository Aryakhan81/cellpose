"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime

from . import transforms, io, dynamics, utils


def batchconv(in_channels, out_channels, sz, conv_3D=False):
    #print(f"batchconv input channels: {in_channels}")
    #print(f"batchconv output channels: {out_channels}")
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        nn.ReLU(inplace=True),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz, conv_3D=False):
    #print(f"batchconv0 input channels: {in_channels}")
    #print(f"batchconv0 output channels: {out_channels}")
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


# Downsampling component
class resdown(nn.Module):

    def __init__(self, in_channels, out_channels, sz, conv_3D=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1, conv_3D)  # use batchnorm0 as "proj"
        for t in range(4):
            if t == 0:  # first layer
                self.conv.add_module("conv_%d" % t,
                                     batchconv(in_channels, out_channels, sz, conv_3D))
            else:  # every other layer
                self.conv.add_module("conv_%d" % t,
                                     batchconv(out_channels, out_channels, sz, conv_3D))

    def forward(self, x):  # forward pass
        # x = self.proj(x) --> just for trolls -- and it works!
        # print("before:")
        # print(x.shape)
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        # print("after:")
        # print(x.shape)
        return x


class downsample(nn.Module):  # class comibining many downsample units

    def __init__(self, nbase, sz, conv_3D=False, max_pool=True):
        super().__init__()
        self.down = nn.Sequential()
        #print(f"nbase: {nbase}")
        if max_pool:
            self.maxpool = nn.MaxPool3d(2, stride=2) if conv_3D else nn.MaxPool2d(
                2, stride=2)
        else:
            self.maxpool = nn.AvgPool3d(2, stride=2) if conv_3D else nn.AvgPool2d(
                2, stride=2)
        for n in range(len(nbase) - 1):
            self.down.add_module("res_down_%d" % n,
                                 resdown(nbase[n], nbase[n + 1], sz, conv_3D))
        
        # define a variable telling us if we have the full count
        self.num_images = 0

    # def forward(self, x):
    #     xd = []
    #     # print("length of slef.down")
    #     # print(len(self.down))
    #     # print(f"shape of x: {x.shape}")

    #     # see how big we are
    #     self.num_images = max(self.num_images, x.shape[0])

    #     # only perform mixing steps when the full set is present
    #     if self.num_images == x.shape[0]: 
    #         split = int(x.shape[0] / 2)
    #         # print(f"split: {str(split)}")

    #         x_1 = x[:split, :, :, :]
    #         x_2 = x[split:, :, :, :]

    #         # # average curr and next frames
    #         # x = torch.add(x_1, x_2)
    #         # x.div_(2)

    #         # # find the max of curr and next frames
    #         # x = torch.max(x_1, x_2)

    #         # find the min of curr and next frames
    #         x = torch.min(x_1, x_2)

    #         # # multiply the two tensors
    #         # x = torch.mul(x_1, x_2)

    #     for n in range(len(self.down)):
    #         if n > 0:
    #             y = self.maxpool(xd[n - 1])
    #         else:
    #             y = x
    #         xd.append(self.down[n](y))
        
    #     # print("len xd:")
    #     # print(len(xd))
    #     return xd
    
    def forward(self, x):
        xd = []

        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))

        xd[-1] = self.min_mix(xd[-1])  # this line is the combination function
        return xd
     
    def avg_mix(self, x):
        self.num_images = max(self.num_images, x.shape[0])
        if self.num_images != x.shape[0]: 
            return x
        
        split = int(x.shape[0] / 2)
        x_1 = x[:split, :, :, :]
        x_2 = x[split:, :, :, :]
        x = torch.add(x_1, x_2)
        x.div_(2)
        x = torch.concatenate([x, x_2], axis=0)
        return x
    
    def max_mix(self, x):
        self.num_images = max(self.num_images, x.shape[0])
        if self.num_images != x.shape[0]: 
            return x
    
        split = int(x.shape[0] / 2)
        x_1 = x[:split, :, :, :]
        x_2 = x[split:, :, :, :]
        x = torch.max(x_1, x_2)
        x = torch.concatenate([x, x_2], axis=0)
        return x
    
    def min_mix(self, x):
        self.num_images = max(self.num_images, x.shape[0])
        if self.num_images != x.shape[0]: 
            return x
    
        split = int(x.shape[0] / 2)
        x_1 = x[:split, :, :, :]
        x_2 = x[split:, :, :, :]
        x = torch.min(x_1, x_2)
        x = torch.concatenate([x, x_2], axis=0)
        return x
    
    def mul_mix(self, x):
        self.num_images = max(self.num_images, x.shape[0])
        if self.num_images != x.shape[0]: 
            return x
    
        split = int(x.shape[0] / 2)
        x_1 = x[:split, :, :, :]
        x_2 = x[split:, :, :, :]
        x = torch.mul(x_1, x_2)
        x = torch.concatenate([x, x_2], axis=0)
        return x


class batchconvstyle(nn.Module):  # for upsampling include style channels

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        super().__init__()
        self.concatenation = False
        # get batchconv just like downsampling
        self.conv = batchconv(in_channels, out_channels, sz, conv_3D)
        # add a linear layer with the "styles"
        self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        #print(f"style: {style.shape}")
        if y is not None:
            x = x + y
        feat = self.full(style)
        #print(f"feat: {feat.shape}")
        for k in range(len(x.shape[2:])):
            feat = feat.unsqueeze(-1)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y


class resup(nn.Module):

    def __init__(self, in_channels, out_channels, style_channels, sz,
                 conv_3D=False):
        super().__init__()
        self.concatenation = False
        self.conv = nn.Sequential()
        self.conv.add_module("conv_0",
                             batchconv(in_channels, out_channels, sz, conv_3D=conv_3D))
        self.conv.add_module(
            "conv_1",
            batchconvstyle(out_channels, out_channels, style_channels, sz,
                           conv_3D=conv_3D))
        self.conv.add_module(
            "conv_2",
            batchconvstyle(out_channels, out_channels, style_channels, sz,
                           conv_3D=conv_3D))
        self.conv.add_module(
            "conv_3",
            batchconvstyle(out_channels, out_channels, style_channels, sz,
                           conv_3D=conv_3D))
        self.proj = batchconv0(in_channels, out_channels, 1, conv_3D=conv_3D)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn),
                             mkldnn=mkldnn)
        return x


class make_style(nn.Module):

    def __init__(self, conv_3D=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool3d if conv_3D else F.avg_pool2d

    def forward(self, x0):
        style = self.avg_pool(x0, kernel_size=x0.shape[2:])
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5
        return style


class upsample(nn.Module):

    def __init__(self, nbase, sz, conv_3D=False):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            self.up.add_module("res_up_%d" % (n - 1),
                               resup(nbase[n], nbase[n - 1], nbase[-1], sz, conv_3D))

    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up) - 2, -1, -1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x


class CPnet(nn.Module):
    """
    CPnet is the Cellpose neural network model used for cell segmentation and image restoration.

    Args:
        nbase (list): List of integers representing the number of channels in each layer of the downsample path.
        nout (int): Number of output channels.
        sz (int): Size of the input image.
        mkldnn (bool, optional): Whether to use MKL-DNN acceleration. Defaults to False.
        conv_3D (bool, optional): Whether to use 3D convolution. Defaults to False.
        max_pool (bool, optional): Whether to use max pooling. Defaults to True.
        diam_mean (float, optional): Mean diameter of the cells. Defaults to 30.0.

    Attributes:
        nbase (list): List of integers representing the number of channels in each layer of the downsample path.
        nout (int): Number of output channels.
        sz (int): Size of the input image.
        residual_on (bool): Whether to use residual connections.
        style_on (bool): Whether to use style transfer.
        concatenation (bool): Whether to use concatenation.
        conv_3D (bool): Whether to use 3D convolution.
        mkldnn (bool): Whether to use MKL-DNN acceleration.
        downsample (nn.Module): Downsample blocks of the network.
        upsample (nn.Module): Upsample blocks of the network.
        make_style (nn.Module): Style module, avgpool's over all spatial positions.
        output (nn.Module): Output module - batchconv layer.
        diam_mean (nn.Parameter): Parameter representing the mean diameter to which the cells are rescaled to during training.
        diam_labels (nn.Parameter): Parameter representing the mean diameter of the cells in the training set (before rescaling).

    """

    def __init__(self, nbase, nout, sz, mkldnn=False, conv_3D=False, max_pool=True,
                 diam_mean=30.):
        super().__init__()
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = True
        self.style_on = True
        self.concatenation = False
        self.conv_3D = conv_3D
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, conv_3D=conv_3D, max_pool=max_pool)
        #self.downsample_1 = downsample(nbase, sz, conv_3D=conv_3D, max_pool=max_pool)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, conv_3D=conv_3D)
        self.make_style = make_style(conv_3D=conv_3D)
        self.output = batchconv(nbaseup[0], nout, 1, conv_3D=conv_3D)
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean,
                                      requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean,
                                        requires_grad=False)
        
        # define boolean to switch off writing to a file to test differences in T0
        self.should_write = False

    @property
    def device(self):
        """
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return next(self.parameters()).device

    def forward(self, data):


        # data dim: Z x channels x X x Y
        # T0 dim: 2D list with X and Y kernel
        """
        Forward pass of the CPnet model.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            tuple: A tuple containing the output tensor, style tensor, and downsampled tensors.
        """
        if self.mkldnn:
            data = data.to_mkldnn()

        # print(f"data shape: {data.shape}")
        T0 = self.downsample(data)  # results of the downsample: contains a list of tensors, as the result of each layer

        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T1 = self.upsample(style, T0, self.mkldnn)
        T1 = self.output(T1)
        if self.mkldnn:
            T0 = [t0.to_dense() for t0 in T0]
            T1 = T1.to_dense()

        if self.should_write:
            #print(data.shape)
            print(f"T0 len: {len(T0)}")
            print(f"T0[-1] shape: {T0[-1].shape}")

            # T0_to_write = T0[0][0][2].detach()
            # data_to_write = data[0][1]
            # print(T0_to_write)
            self.should_write = False
            # np.savetxt('/Users/aryagharib/Desktop/T0_img1', T0_to_write)
            # #np.savetxt('/Users/aryagharib/Desktop/erf2.txt', T0[0])

            #print(T0[0][0][0])
            # torch.save(T0[0], "/Users/aryagharib/Desktop/T0_0_img3.pt")
            # torch.save(T0[1], "/Users/aryagharib/Desktop/T0_1_img3.pt")
            # torch.save(T0[2], "/Users/aryagharib/Desktop/T0_2_img3.pt")
            # torch.save(T0[3], "/Users/aryagharib/Desktop/T0_3_img3.pt")

            # torch.save(T1[0], "/Users/aryagharib/Desktop/T1_0_img3.pt")
            # torch.save(T1[1], "/Users/aryagharib/Desktop/T1_1_img3.pt")
            # torch.save(T1[2], "/Users/aryagharib/Desktop/T1_2_img3.pt")
            # torch.save(T1[3], "/Users/aryagharib/Desktop/T1_3_img3.pt")

        return T1, style0, T0

    def save_model(self, filename):
        """
        Save the model to a file.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device=None):
        """
        Load the model from a file.

        Args:
            filename (str): The path to the file where the model is saved.
            device (torch.device, optional): The device to load the model on. Defaults to None.
        """
        if (device is not None) and (device.type != "cpu"):
            state_dict = torch.load(filename, map_location=device)
        else:
            self.__init__(self.nbase, self.nout, self.sz, self.mkldnn, self.conv_3D,
                          self.diam_mean)
            state_dict = torch.load(filename, map_location=torch.device("cpu"))

        if state_dict["output.2.weight"].shape[0] != self.nout:
            for name in self.state_dict():
                if "output" not in name:
                    self.state_dict()[name].copy_(state_dict[name])
        else:
            self.load_state_dict(
                dict([(name, param) for name, param in state_dict.items()]),
                strict=False)

# if __name__ == '__main__':
#     data = 
#     forward(data)
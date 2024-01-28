import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F

from collections import OrderedDict


class RubikCube_multiply(nn.Module):
    def __init__(self, nc, out, shiftPixel=1, gc=4):
        super(RubikCube_multiply, self).__init__()

        self.processC1 = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.processC2 = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.processC3 = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.processC4 = nn.Sequential(
            nn.Conv2d(gc, gc, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.processOutput = nn.Sequential(
            nn.Conv2d(nc, out, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.shiftPixel = shiftPixel
        self.gc = gc
        self.split_indexes = (gc, gc, gc, gc, nc - gc * 4)

    def shift_feat(self, x, shiftPixel, g):
        B, C, H, W = x.shape
        out = torch.zeros_like(x)

        out[:, g * 0:g * 1, :, :-shiftPixel] = x[:, g * 0:g * 1, :, shiftPixel:]  # shift left
        out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right
        out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up
        out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down

        out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift
        return out

    def forward(self, x):
        residual = x
        x_shifted = self.shift_feat(x, self.shiftPixel, self.gc)
        c1, c2, c3, c4, x2 = torch.split(x_shifted, self.split_indexes, dim=1)

        c1_processed = self.processC1(c1)
        c2_processed = self.processC2(c1_processed * c2)
        c3_processed = self.processC3(c2_processed * c3)
        c4_processed = self.processC4(c3_processed * c4)

        out = torch.cat([c1_processed, c2_processed, c3_processed, c4_processed, x2], dim=1)

        return self.processOutput(out) + residual


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


@ARCH_REGISTRY.register()
class DNCNN_RubikConv(nn.Module):
    def __init__(self, in_nc=1, out_nc=3, nc=64, nb=20, act_mode='BR', shiftPixel=1, gc=4):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DNCNN_RubikConv, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        self.m_head = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)

        # self.m_body1 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body1 = RubikCube_multiply(nc, nc, shiftPixel, gc)
        # self.m_body2 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body2 = RubikCube_multiply(nc, nc, shiftPixel, gc)
        # self.m_body3 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body3 = RubikCube_multiply(nc, nc, shiftPixel, gc)

        self.m_body4 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body5 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body6 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body7 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body8 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body9 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body10 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body11 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body12 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body13 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body14 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body15 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body16 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body17 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_body18 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.m_tail = conv(nc, out_nc, mode='C', bias=bias)


    def forward(self, x):
        n = self.m_head(x)

        
        n = self.m_body1(n)
        n = self.m_body2(n)
        n = self.m_body3(n)
        n = self.m_body4(n)
        n = self.m_body5(n)
        n = self.m_body6(n)
        n = self.m_body7(n)
        n = self.m_body8(n)
        n = self.m_body9(n)
        n = self.m_body10(n)
        n = self.m_body11(n)
        n = self.m_body12(n)
        n = self.m_body13(n)
        n = self.m_body14(n)
        n = self.m_body15(n)
        n = self.m_body16(n)
        n = self.m_body17(n)
        n = self.m_body18(n)
        
        n = self.m_tail(n)
        return x-n


# if __name__ == '__main__':
#     print_network(DnCNN(in_nc=1, out_nc=3, nc=64, nb=30)) # paras: 1039939
#     print_network(DnCNN(in_nc=1, out_nc=3, nc=64, nb=20))  # paras: 669379
#     print_network(DnCNN(in_nc=1, out_nc=3, nc=96, nb=20)) # paras: 1501731



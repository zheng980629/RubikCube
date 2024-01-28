import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F

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


def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)


def default_act():
    return nn.ReLU(True)


class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act):

        super(ResBlock, self).__init__()

        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm: modules.append(norm(n_feats))
            if act and i == 0: modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class ResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feats=None, kernel_size=None, n_resblocks=None, rgb_range=None, mean_shift=True, shiftPixel=1, gc=8):
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feats = n_feats 
        self.kernel_size = kernel_size
        self.n_resblocks = n_resblocks

        self.mean_shift = mean_shift
        self.rgb_range = rgb_range
        self.mean = self.rgb_range / 2

        modules = []
        modules.append(default_conv(self.in_channels, self.n_feats, self.kernel_size))

        for _ in range(2):
            modules.append(ResBlock(self.n_feats, self.kernel_size))
        modules.append(RubikCube_multiply(self.n_feats, self.n_feats, shiftPixel, gc))
        modules.append(RubikCube_multiply(self.n_feats, self.n_feats, shiftPixel, gc))

        for _ in range(self.n_resblocks - 4):
            modules.append(ResBlock(self.n_feats, self.kernel_size))

        modules.append(default_conv(self.n_feats, self.out_channels, self.kernel_size))

        self.body = nn.Sequential(*modules)

    def forward(self, input):
        if self.mean_shift:
            input = input - self.mean

        output = self.body(input)

        if self.mean_shift:
            output = output + self.mean

        return output


class conv_end(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, ratio=2):
        super(conv_end, self).__init__()

        modules = [
            default_conv(in_channels, out_channels, kernel_size),
            nn.PixelShuffle(ratio)
        ]

        self.uppath = nn.Sequential(*modules)

    def forward(self, x):
        return self.uppath(x)


@ARCH_REGISTRY.register()
class deepDeblur_RubikConv(nn.Module):
    def __init__(self, rgb_range=1.0, n_resblocks=19, n_feats=64, n_scales=3, kernel_size=5, shiftPixel=1, gc=8):
        super(deepDeblur_RubikConv, self).__init__()

        self.rgb_range = rgb_range
        self.mean = self.rgb_range / 2

        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.kernel_size = kernel_size

        self.n_scales = n_scales

        self.body_models = nn.ModuleList([
            ResNet(3, 3, rgb_range=self.rgb_range, n_feats=self.n_feats, kernel_size=self.kernel_size, n_resblocks=self.n_resblocks, mean_shift=False, shiftPixel=shiftPixel, gc=gc),
        ])
        for _ in range(1, self.n_scales):
            self.body_models.insert(0, ResNet(6, 3, rgb_range=self.rgb_range, n_feats=self.n_feats, kernel_size=self.kernel_size, n_resblocks=self.n_resblocks, mean_shift=False, shiftPixel=shiftPixel, gc=gc))

        self.conv_end_models = nn.ModuleList([None])
        for _ in range(1, self.n_scales):
            self.conv_end_models += [conv_end(3, 12)]

    def forward(self, x):

        input_pyramid = [x, x[:, :, ::2, ::2], x[:, :, ::4, ::4]]
        
        scales = range(self.n_scales-1, -1, -1)    # 0: fine, 2: coarse

        for s in scales:
            input_pyramid[s] = input_pyramid[s] - self.mean

        output_pyramid = [None] * self.n_scales

        input_s = input_pyramid[-1]
        for s in scales:    # [2, 1, 0]
            output_pyramid[s] = self.body_models[s](input_s)
            if s > 0:
                up_feat = self.conv_end_models[s](output_pyramid[s])
                up_feat = F.interpolate(up_feat, size=input_pyramid[s-1].shape[-2:])
                input_s = torch.cat((input_pyramid[s-1], up_feat), 1)

        for s in scales:
            output_pyramid[s] = output_pyramid[s] + self.mean

        return output_pyramid[0], output_pyramid[1], output_pyramid[2]

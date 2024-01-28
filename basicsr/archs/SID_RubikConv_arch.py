import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer, ConvLReLUNoBN, upsample_and_concat, single_conv, up, outconv
from basicsr.utils.registry import ARCH_REGISTRY


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


@ARCH_REGISTRY.register()
class SID_RubikConv(nn.Module):
    def __init__(self, shiftPixel=1, gc=4):
        super(SID_RubikConv, self).__init__()

        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = RubikCube_multiply(64, 64, shiftPixel, gc)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = RubikCube_multiply(64, 64, shiftPixel, gc)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, 3, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = F.interpolate(up6, size=(conv4.shape[-2:]))
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = F.interpolate(up7, size=(conv3.shape[-2:]))
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = F.interpolate(up8, size=(conv2.shape[-2:]))
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = F.interpolate(up9, size=(conv1.shape[-2:]))
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        out= self.conv10_1(conv9)
        return out

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt

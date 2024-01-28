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


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        feat1 = self.convs(x)
        feat2 = self.LFF(feat1) + x
        return feat2

class DRBN_BU(nn.Module):
    def __init__(self, n_color, shiftPixel):
        super(DRBN_BU, self).__init__()

        G0 = 16
        kSize = 3
        self.D = 6
        G = 8
        C = 4

        self.SFENet1 = nn.Conv2d(n_color*2, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.RDBs = nn.ModuleList()

        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = 2*G0, growRate = 2*G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = 2*G0, growRate = 2*G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.UPNet2 = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.UPNet4 = nn.Sequential(*[
                nn.Conv2d(G0*2, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.Down1 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2)
        self.Down2 = nn.Conv2d(G0, G0*2, kSize, padding=(kSize-1)//2, stride=2)

        self.Up1 = nn.ConvTranspose2d(G0, G0, kSize+1, stride=2, padding=1)
        self.Up2 = nn.ConvTranspose2d(G0*2, G0, kSize+1, stride=2, padding=1)

        self.Relu = nn.ReLU()
        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear')

    def part_forward(self, x):
        #
        # Stage 1
        #
        flag = x[0]
        input_x = x[1]

        prev_s1 = x[2]
        prev_s2 = x[3]
        prev_s4 = x[4]

        prev_feat_s1 = x[5]
        prev_feat_s2 = x[6]
        prev_feat_s4 = x[7]

        f_first = self.Relu(self.SFENet1(input_x))
        f_s1  = self.Relu(self.SFENet2(f_first))
        f_s2 = self.Down1(self.RDBs[0](f_s1))
        f_s4 = self.Down2(self.RDBs[1](f_s2))

        if flag == 0:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4))
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4))
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2))+f_first
        else:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4)) + prev_feat_s4
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4)) + prev_feat_s2
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2))+f_first + prev_feat_s1

        res4 = self.UPNet4(f_s4)
        res2 = self.UPNet2(f_s2) + self.Img_up(res4)
        res1 = self.UPNet(f_s1) + self.Img_up(res2)

        return res1, res2, res4, f_s1, f_s2, f_s4


    def forward(self, x_input):
        x = x_input

        res1, res2, res4, f_s1, f_s2, f_s4 = self.part_forward(x)

        return res1, res2, res4, f_s1, f_s2, f_s4



class DRBN_BU_rubikeCubeIdentityGC(nn.Module):
    def __init__(self, n_color, shiftPixel, gc):
        super(DRBN_BU_rubikeCubeIdentityGC, self).__init__()

        G0 = 16
        kSize = 3
        self.D = 6
        G = 8
        C = 4

        self.SFENet1 = nn.Conv2d(n_color*2, G0, kSize, padding=(kSize-1)//2, stride=1)
        # self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = RubikCube_multiply(G0, G0, shiftPixel=shiftPixel, gc=gc)

        self.RDBs = nn.ModuleList()

        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = 2*G0, growRate = 2*G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = 2*G0, growRate = 2*G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.UPNet2 = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.UPNet4 = nn.Sequential(*[
                nn.Conv2d(G0*2, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.Down1 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2)
        self.Down2 = nn.Conv2d(G0, G0*2, kSize, padding=(kSize-1)//2, stride=2)

        self.Up1 = nn.ConvTranspose2d(G0, G0, kSize+1, stride=2, padding=1)
        self.Up2 = nn.ConvTranspose2d(G0*2, G0, kSize+1, stride=2, padding=1)

        self.Relu = nn.ReLU()
        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear')

    def part_forward(self, x):
        #
        # Stage 1
        #
        flag = x[0]
        input_x = x[1]

        prev_s1 = x[2]
        prev_s2 = x[3]
        prev_s4 = x[4]

        prev_feat_s1 = x[5]
        prev_feat_s2 = x[6]
        prev_feat_s4 = x[7]

        f_first = self.Relu(self.SFENet1(input_x))
        f_s1  = self.Relu(self.SFENet2(f_first))
        f_s2 = self.Down1(self.RDBs[0](f_s1))
        f_s4 = self.Down2(self.RDBs[1](f_s2))

        if flag == 0:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4))
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4))
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2))+f_first
        else:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4)) + prev_feat_s4
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4)) + prev_feat_s2
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2))+f_first + prev_feat_s1

        res4 = self.UPNet4(f_s4)
        res2 = self.UPNet2(f_s2) + self.Img_up(res4)
        res1 = self.UPNet(f_s1) + self.Img_up(res2)

        return res1, res2, res4, f_s1, f_s2, f_s4


    def forward(self, x_input):
        x = x_input

        res1, res2, res4, f_s1, f_s2, f_s4 = self.part_forward(x)

        return res1, res2, res4, f_s1, f_s2, f_s4


@ARCH_REGISTRY.register()
class DRBN_RubikConv(nn.Module):
    def __init__(self, n_color, shiftPixel=1, gc=3):
        super(DRBN_RubikConv, self).__init__()

        self.recur1 = DRBN_BU_rubikeCubeIdentityGC(n_color, shiftPixel, gc)
        self.recur2 = DRBN_BU(n_color, shiftPixel)
        self.recur3 = DRBN_BU(n_color, shiftPixel)
        self.recur4 = DRBN_BU(n_color, shiftPixel)

    def forward(self, x_input):
        x = x_input

        res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4 = self.recur1([0, torch.cat((x, x), 1), 0, 0, 0, 0, 0, 0])
        res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4 = self.recur2([1, torch.cat((res_g1_s1, x), 1), res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4])
        res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.recur3([1, torch.cat((res_g2_s1, x), 1), res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4])
        res_g4_s1, res_g4_s2, res_g4_s4, feat_g4_s1, feat_g4_s2, feat_g4_s4 = self.recur4([1, torch.cat((res_g3_s1, x), 1), res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4])

        return res_g4_s1, res_g4_s2, res_g4_s4

# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# contrast-aware channel attention block
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

# Texture extraction using sobel operator
class Gradient_conv(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Gradient_conv, self).__init__()
        kernel = np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])
        self.convx = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=in_channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(kernel))
        self.convy = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=in_channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(kernel.T))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)  # B C_in H W
        attention = self.attention(x)  # B 1 H W
        x = attention * x  # B C_in H W
        out = self.conv1x1(x)  # B C_out H W
        return out


class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

# Texture gradient extraction module
class EDBB_deploy(nn.Module):
    def __init__(self, inp_planes, out_planes):
        super(EDBB_deploy, self).__init__()

        self.rep_conv = Gradient_conv(in_channels=inp_planes, out_channels=out_planes)
        self.act = nn.PReLU(num_parameters=out_planes)
        self.conv = nn.Conv2d(in_channels=inp_planes, out_channels=out_planes, kernel_size=1)

    def forward(self, x):
        y = self.rep_conv(x)
        # y = self.act(y)
        # y = self.conv(y)

        return y

# Recovery and reconstruction module
class EDBB_rec(nn.Module):
    def __init__(self, inp_planes, out_planes):
        super(EDBB_rec, self).__init__()

        self.rep_conv = nn.Conv2d(in_channels=inp_planes, out_channels=out_planes, kernel_size=3, stride=1,
                                  padding=1)

        self.act = nn.PReLU(num_parameters=out_planes)

    def forward(self, x):
        y = self.rep_conv(x)
        y = self.act(y)

        return y

# enhanced spatial attention block
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

# Attention selection module
class nir_attention(nn.Module):
    def __init__(self, n_feats=64, conv=nn.Conv2d):
        super(nir_attention, self).__init__()
        self.eca = ESA(n_feats, conv)
        self.cca = CCALayer(n_feats)

    def forward(self, x):
        out = self.eca(x)
        out = self.cca(out)

        return out

# Attention selection module
class vis_attention(nn.Module):
    def __init__(self, n_feats=64, conv=nn.Conv2d):
        super(vis_attention, self).__init__()
        self.eca = ESA(n_feats, conv)
        self.cca = CCALayer(n_feats)

    def forward(self, x):
        out = self.eca(x)
        out = self.cca(out)

        return out


class conv(nn.Module):
    def __init__(self, n_feats):
        super(conv, self).__init__()
        self.conv1x1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.act = nn.PReLU(num_parameters=n_feats)

    def forward(self, x):
        return self.act(self.conv1x1(x))

# NIR image texture edge extraction module
class nir_Cell(nn.Module):
    def __init__(self, n_feats=48):
        super(nir_Cell, self).__init__()

        self.conv1 = conv(n_feats)  # nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.conv2 = EDBB_deploy(n_feats, n_feats)
        self.conv3 = EDBB_deploy(n_feats, n_feats)

        self.fuse = nn.Conv2d(n_feats * 2, n_feats, 1, 1, 0)

        self.att = ESA(n_feats, nn.Conv2d)  # MAB(n_feats)# ENLCA(n_feats)  #CoordAtt(n_feats,n_feats,10)#

        self.branch = nn.ModuleList([nn.Conv2d(n_feats, n_feats // 2, 1, 1, 0) for _ in range(4)])

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        # fuse [x, out1, out2, out3]
        out = self.fuse(
            torch.cat([self.branch[0](x), self.branch[1](out1), self.branch[2](out2), self.branch[3](out3)], dim=1))
        out = self.att(out)
        out += x

        return out

# VIS image texture edge extraction module
class vis_Cell(nn.Module):
    def __init__(self, n_feats=48):
        super(vis_Cell, self).__init__()

        self.conv1 = conv(n_feats)  # nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.conv2 = EDBB_deploy(n_feats, n_feats)
        self.conv3 = EDBB_deploy(n_feats, n_feats)

        self.fuse = nn.Conv2d(n_feats * 2, n_feats, 1, 1, 0)

        self.att = ESA(n_feats, nn.Conv2d)

        self.branch = nn.ModuleList([nn.Conv2d(n_feats, n_feats // 2, 1, 1, 0) for _ in range(4)])

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        # fuse [x, out1, out2, out3]
        out = self.fuse(
            torch.cat([self.branch[0](x), self.branch[1](out1), self.branch[2](out2), self.branch[3](out3)], dim=1))
        out = self.att(out)
        out += x

        return out

# reconstruction module
class rec_Cell(nn.Module):
    def __init__(self, n_feats=48):
        super(rec_Cell, self).__init__()

        self.conv1 = conv(n_feats)
        self.conv2 = EDBB_rec(n_feats, n_feats)
        self.conv3 = EDBB_rec(n_feats, n_feats)
        self.fuse = nn.Conv2d(n_feats * 2, n_feats, 1, 1, 0)
        # self.att = ESA(n_feats, nn.Conv2d)
        self.branch = nn.ModuleList([nn.Conv2d(n_feats, n_feats // 2, 1, 1, 0) for _ in range(4)])

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out = self.fuse(
            torch.cat([self.branch[0](x), self.branch[1](out1), self.branch[2](out2), self.branch[3](out3)], dim=1))
        # out = self.att(out)
        out += x
        return out

# The whole fusion network
class MyNet(nn.Module):
    def __init__(self, in_channels=1, n_feats=64, out_channels=1):
        super(MyNet, self).__init__()
        self.nir_head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)
        self.vis_head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)

        # body cells
        self.vis_cells = nn.ModuleList([vis_Cell(n_feats) for i in range(4)])
        self.nir_cells = nn.ModuleList([nir_Cell(n_feats) for i in range(4)])

        # attention branch
        self.nir_Attention = nn.ModuleList([nir_attention(n_feats, nn.Conv2d) for _ in range(4)])
        self.vis_Attention = nn.ModuleList([vis_attention(n_feats, nn.Conv2d) for _ in range(4)])

        # fusion
        self.cat = nn.Conv2d(n_feats * 2, n_feats, 1)

        # reconstruction
        self.rec_cells = nn.ModuleList([rec_Cell(n_feats) for _ in range(4)])
        self.reconstruction = nn.ModuleList([nn.Conv2d(n_feats * 3, n_feats, 1, 1, 0) for _ in range(4)])
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, out_channels, 3, 1, 1),
        )

    def forward(self, vis_image, nir_image):
        # nir extractor
        nir_out0 = self.nir_head(nir_image)
        nir_out1 = self.nir_cells[0](nir_out0)
        nir_out2 = self.nir_cells[1](nir_out1)
        nir_out3 = self.nir_cells[2](nir_out2)
        nir_out4 = self.nir_cells[3](nir_out3)

        # nir extractor
        vis_out0 = self.vis_head(vis_image)
        vis_out1 = self.vis_cells[0](vis_out0)
        vis_out2 = self.vis_cells[1](vis_out1)
        vis_out3 = self.vis_cells[2](vis_out2)
        vis_out4 = self.vis_cells[3](vis_out3)

        # concat
        concat = torch.cat([nir_out4, vis_out4], dim=1)
        out4_fuse = self.cat(concat)

        # reconstruction
        out4 = self.rec_cells[3](out4_fuse)

        attention_nir_out3 = self.nir_Attention[3](nir_out3)
        attention_vis_out3 = self.vis_Attention[3](vis_out3)
        out3_fuse = self.reconstruction[3](torch.cat([out4, attention_nir_out3, attention_vis_out3], dim=1))
        out3 = self.rec_cells[2](out3_fuse)

        attention_nir_out2 = self.nir_Attention[2](nir_out2)
        attention_vis_out2 = self.vis_Attention[2](vis_out2)
        out2_fuse = self.reconstruction[2](torch.cat([out3, attention_nir_out2, attention_vis_out2], dim=1))
        out2 = self.rec_cells[1](out2_fuse)

        attention_nir_out1 = self.nir_Attention[1](nir_out1)
        attention_vis_out1 = self.vis_Attention[1](vis_out1)
        out1_fuse = self.reconstruction[1](torch.cat([out2, attention_nir_out1, attention_vis_out1], dim=1))
        out1 = self.rec_cells[0](out1_fuse)

        attention_nir_out0 = self.nir_Attention[0](nir_out0)
        attention_vis_out0 = self.vis_Attention[0](vis_out0)
        out1_fuse = self.reconstruction[0](torch.cat([out1, attention_nir_out0, attention_vis_out0], dim=1))
        out = self.tail(out1_fuse)

        return out


if __name__ == '__main__':
    model = MyNet()
    x = torch.randn(1, 1, 100, 100)
    y = torch.randn(1, 1, 100, 100)
    print(model)
    b = model(x, x)
    print(x.shape, b.shape)

    # writer = SummaryWriter("logs")
    # writer.add_graph(model, (x, y))
    # writer.close()

# tensorboard --logdir=C:\Users\ph\PycharmProjects\FakeMotionBlur-master\logs --port=6007

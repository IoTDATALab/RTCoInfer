# -*- coding:utf-8 -*-
import math
import torch.nn as nn
import torch as t
import time
import numbits

from .SWCNN_ops import SWBatchNorm2d, SWConv2d, make_divisible,SWBatchNorm2d_2
from utils.config import FLAGS
#zzh:no change on input/output
class InvertedResidual0(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual0, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            layers += [
                SWConv2d(
                    inp, expand_inp, 1, 1, 0, bias=False,us=[False,False],
                    ratio=[1, expand_ratio]),
                nn.BatchNorm2d(expand_inp),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            SWConv2d(
                expand_inp, expand_inp, 3, stride, 1, groups=expand_inp,
                depthwise=True, bias=False,us=[False,False],
                ratio=[expand_ratio, expand_ratio]),
            nn.BatchNorm2d(expand_inp),

            nn.ReLU6(inplace=True),

            SWConv2d(
                expand_inp, outp, 1, 1, 0, bias=False,us=[False,False],
                ratio=[expand_ratio, 1]),
            nn.BatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res

#zzh:change on input
class InvertedResidual1(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual1, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp * expand_ratio
        # if expand_ratio != 1:
            # layers += [
                # SWConv2d(
                    # inp, expand_inp, 1, 1, 0, bias=False,us=[True,False],
                    # ratio=[1, expand_ratio]),
                # SWBatchNorm2d_2(expand_inp, ratio=expand_ratio),
                # nn.ReLU6(inplace=True),
            # ]
        # depthwise + project back
        layers += [
            SWConv2d(
                expand_inp, expand_inp, 3, stride, 1, groups=1,
                depthwise=False, bias=False,us=[True,False],
                ratio=[expand_ratio, expand_ratio]),
            SWBatchNorm2d_2(expand_inp, ratio=expand_ratio),

            nn.ReLU6(inplace=True),

            SWConv2d(
                expand_inp, outp, 1, 1, 0, bias=False,us=[False,False],
                ratio=[expand_ratio, 1]),
            SWBatchNorm2d_2(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res

#zzh:change on output
class InvertedResidual2(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual2, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            layers += [
                SWConv2d(
                    inp, expand_inp, 1, 1, 0, bias=False,us=[False,False],
                    ratio=[1, expand_ratio]),
                nn.BatchNorm2d(expand_inp),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            SWConv2d(
                expand_inp, expand_inp, 3, stride, 1, groups=expand_inp,
                depthwise=True, bias=False,us=[False,False],
                ratio=[expand_ratio, expand_ratio]),
            nn.BatchNorm2d(expand_inp),

            nn.ReLU6(inplace=True),

            SWConv2d(
                expand_inp, outp, 1, 1, 0, bias=False,us=[False,True],
                ratio=[expand_ratio, 1]),
            SWBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res

#zzh:no change on input/output but only change BN layers
class InvertedResidual3(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual3, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            layers += [
                SWConv2d(
                    inp, expand_inp, 1, 1, 0, bias=False,us=[False,False],
                    ratio=[1, expand_ratio]),
                SWBatchNorm2d_2(expand_inp, ratio=expand_ratio),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            SWConv2d(
                expand_inp, expand_inp, 3, stride, 1, groups=expand_inp,
                depthwise=True, bias=False,us=[False,False],
                ratio=[expand_ratio, expand_ratio]),
            SWBatchNorm2d_2(expand_inp, ratio=expand_ratio),

            nn.ReLU6(inplace=True),

            SWConv2d(
                expand_inp, outp, 1, 1, 0, bias=False,us=[False,False],
                ratio=[expand_ratio, 1]),
            SWBatchNorm2d_2(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()
        self.quant_bits=255

        # setting of inverted residual blocks
        self.block_setting_part1 = [
            # t, c, n, s
            [1, 16, 1, 1],
        ]
        self.block_setting_part2 = [
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = []

        width_mult = 1.0
        # head
        assert input_size % 32 == 0
        channels = make_divisible(32 * width_mult)
        self.outp = int(1280)
        first_stride = 2
        self.features.append(
            nn.Sequential(
                SWConv2d(
                    3, channels, 3, stride=first_stride, padding=1, bias=False,
                    us=[False, True]),
                SWBatchNorm2d(channels),
                nn.ReLU6(inplace=False))
        )
        

        # zzh:body1,train the part1
        for t, c, n, s in self.block_setting_part1:
            if [t, c, n, s]==[1, 16, 1, 1]:
                outp = int(c)
                for i in range(n):
                    if i == 0:
                        self.features.append(
                            InvertedResidual1(channels, outp, s, t))
                    channels=outp
        #zzh:body2,train the part2
        for t, c, n, s in self.block_setting_part2:
            outp = int(c)
            if [t, c, n, s] == self.block_setting_part2[0]:
                for i in range(n):
                    if i == 0:
                        self.features.append(
                            InvertedResidual3(channels, outp, s, t))
                    else:
                        self.features.append(
                            InvertedResidual3(channels, outp, 1, t))
                    channels = outp
            else:
                for i in range(n):
                    if i == 0:
                        self.features.append(
                            InvertedResidual3(channels, outp, s, t))
                    else:
                        self.features.append(
                            InvertedResidual3(channels, outp, 1, t))
                    channels = outp

        # tail
        self.features.append(
            nn.Sequential(
                SWConv2d(
                    channels, self.outp, 1, 1, 0, bias=False,
                    us=[False, False]),
                SWBatchNorm2d_2(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(self.outp, num_classes))
        if FLAGS.reset_parameters:
            self.reset_parameters()          

    def forward(self, x):
#        f = open('./linshi.txt','a')
        x = self.features[0](x)
        _x=x.detach()
        for _ in range(_x.shape[1]):
            _max=t.max(_x[:,_])
            _min=t.min(_x[:,_])         
            code_book=t.round((_x[:,_]-_min)*self.quant_bits/(_max-_min))
            _x[:,_]=_min+code_book*((_max-_min)/self.quant_bits)
        x.data=_x
        x = self.features[1:](x)
        x = x.view(-1, self.outp)
        x = self.classifier(x)
#        f.close()
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

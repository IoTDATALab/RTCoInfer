# -*- coding:utf-8 -*-
import torch.nn as nn


from utils.config import FLAGS


def make_divisible(v, divisor=1, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SWConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1], preserve_ratio=1.0):
        super(SWConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = preserve_ratio
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        if getattr(FLAGS, 'conv_averaged', False):
            y = y * (max(self.in_channels_list) / self.in_channels)
        return y



class SWBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        super(SWBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=True)
        self.num_features_max = num_features
        a=FLAGS.num_list
        b=FLAGS.quant_list
        zzh_bn=[]
        for i in range(len(a)):
            for j in range(len(b)):
                zzh_bn.append(int(self.num_features_max*a[i]))
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(i, affine=True) for i in zzh_bn])
        self.ratio = ratio
        self.width_mult = None
        self.quant_bits=None
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c = make_divisible(
            self.num_features_max * self.width_mult / self.ratio) * self.ratio
        idx = int(FLAGS.num_list.index(self.width_mult)*len(FLAGS.quant_list)+FLAGS.quant_list.index(self.quant_bits))
        y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y

#Do not modify the output numbers 'c', but create 'c' bn layers with same output_size for diferent call;
class SWBatchNorm2d_2(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        super(SWBatchNorm2d_2, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        # for tracking performance during training
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(self.num_features_max, affine=False) for i in range(len(FLAGS.num_list)*len(FLAGS.quant_list))])
        self.ratio = ratio
        self.width_mult = None
        self.quant_bits=None
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        idx = int(FLAGS.num_list.index(self.width_mult)*len(FLAGS.quant_list)+FLAGS.quant_list.index(self.quant_bits))
        y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean,
                self.bn[idx].running_var,
                weight,
                bias,
                self.training,
                self.momentum,
                self.eps)
        return y


def pop_channels(autoslim_channels):
    return [i.pop(0) for i in autoslim_channels]

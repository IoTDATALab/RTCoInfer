import torch
import torch.nn as nn

from einops import rearrange
from utils.config import FLAGS

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        # nn.BatchNorm2d(oup),
        SWBatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        # nn.BatchNorm2d(oup),
        SWBatchNorm2d(oup),
        nn.SiLU()
    )

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

def Quant(detach_x,quant_bits):
    _x = detach_x
    for _ in range(_x.shape[1]):
        _max=torch.max(_x[:,_])
        _min=torch.min(_x[:,_])         
        code_book=torch.round((_x[:,_]-_min)*quant_bits/(_max-_min))
        _x[:,_]=_min+code_book*((_max-_min)/quant_bits)
    return _x


class SWLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, us = False):
        super(SWLayerNorm, self).__init__(normalized_shape, eps=1e-5)

        self.num_features_max = normalized_shape
        a=FLAGS.num_list
        b=FLAGS.quant_list
        zzh_ln=[]
        for i in range(len(a)):
            for j in range(len(b)):
                if us:
                    zzh_ln.append(int(self.num_features_max*a[i]))
                else:
                    zzh_ln.append(int(self.num_features_max))
        
        self.ln = nn.ModuleList([nn.LayerNorm(norm_shape,elementwise_affine=True) for norm_shape in zzh_ln])

        self.width_mult = None
        self.quant_bits=None

    def forward(self, input):
        idx = int(FLAGS.num_list.index(self.width_mult)*len(FLAGS.quant_list)+FLAGS.quant_list.index(self.quant_bits))

        return nn.functional.layer_norm(
            input, 
            self.normalized_shape, 
            self.ln[idx].weight, 
            self.ln[idx].bias, 
            self.eps)

class SWBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1,us = False):
        super(SWBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=True)
        self.num_features_max = num_features
        a=FLAGS.num_list
        b=FLAGS.quant_list
        zzh_bn=[]

        for i in range(len(a)):
            for j in range(len(b)):
                if us:
                    zzh_bn.append(int(self.num_features_max*a[i]))
                else:
                    zzh_bn.append(int(self.num_features_max))

        self.bn = nn.ModuleList([
            nn.BatchNorm2d(i, affine=True) for i in zzh_bn])
        self.ratio = ratio
        self.width_mult = None
        self.quant_bits=None
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = int(FLAGS.num_list.index(self.width_mult)*len(FLAGS.quant_list)+FLAGS.quant_list.index(self.quant_bits))
        y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean,
                self.bn[idx].running_var,
                self.bn[idx].weight,
                self.bn[idx].bias,
                self.training,
                self.momentum,
                self.eps)
        return y

class SWConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[False, False], ratio=[1, 1]):
        super(SWConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
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
        return y

def SWconv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        # nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        SWConv2d(inp, oup, kernal_size, stride, 1, bias=False,us = [False,True]),
        # nn.BatchNorm2d(oup),
        SWBatchNorm2d(oup,us=True),
        nn.SiLU()
    )


class SWMV2Block(nn.Module):
    # xs or s
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
 
        self.conv = nn.Sequential(
            # pw
            # nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            SWConv2d(inp, hidden_dim, 1, 1, 0, bias=False,us = [True,False]),
            # nn.BatchNorm2d(hidden_dim),
            SWBatchNorm2d(hidden_dim),
            nn.SiLU(),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            SWBatchNorm2d(hidden_dim),
            nn.SiLU(),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(oup),
            SWBatchNorm2d(oup),
        )

    def forward(self, x):
        return self.conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.norm = SWLayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                SWBatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup),
                SWBatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                SWBatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                SWBatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(oup),
                SWBatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class Model(nn.Module):
# dims = [144, 192, 240]
# channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
# return MobileViT((224, 224), dims, channels, num_classes=1000)
# class MobileViT(nn.Module):
    def __init__(self, num_classes = 1000, input_size = 224, dims = [144, 192, 240], channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640], expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        self.quant_bits = None
        ih = input_size
        iw = input_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        # self.conv1 = conv_nxn_bn(3, int(channels[0]), stride=2)
        self.SWconv1 = SWconv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        # self.mv2.append(MV2Block(int(channels[0]), channels[1], 1, expansion))
        self.mv2.append(SWMV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih//32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        # x = self.conv1(x)
        x = self.SWconv1(x)
        x.data = Quant(detach_x=x.detach(),quant_bits=self.quant_bits)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((224, 224), dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(5, 3, 256, 256)
    # vit = mobilevit_xxs()
    # out = vit(img)
    # print(out.shape)
    # print(count_parameters(vit))

    # vit = mobilevit_xs()
    # out = vit(img)
    # print(out.shape)
    # print(count_parameters(vit))

    # vit = mobilevit_s()
    # out = vit(img)
    # print(out.shape)
    # print(count_parameters(vit))

    data = torch.randn(5, 3, 256, 256)
    model = mobilevit_s()

    output = model(data)
    with open('v2_model.txt','a+') as f:
        f.write(str(model))
    # print(model)
    onnx_path = "onnx_model_vit.onnx"
    torch.onnx.export(model, data, onnx_path)
    import netron
    netron.start(onnx_path)
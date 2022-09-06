'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mobilenetv2 import mobilenet_v2
from .utils import process_state_dict
import copy
import pickle
import sys
import numbits
import numpy as np

__all__ = ['CLIOMobileNetV2']

class CLIOMobileNetV2(nn.Module):

    def __init__(self, split_point=0, widths=[1,2,3,4,5,6,7,8,16,32], use_random_connect=True, num_classes=196, pretrained=False, reload_path=None):
        super(CLIOMobileNetV2, self).__init__()

        self.widths = widths
        self.default_width = 32
        self.use_random_connect = use_random_connect

        net = mobilenet_v2(pretrained=pretrained)

        num_ftrs = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(num_ftrs, num_classes)

        if reload_path:
            checkpoint = torch.load(reload_path)
            net.load_state_dict(process_state_dict(checkpoint['state_dict']))
        
        self.shared_layers = net.features[0:split_point+1]
        self.heads = nn.ModuleDict()
        self.linears = nn.ModuleDict()
        in_channels = 32
        for w in self.widths:
            base_net = copy.deepcopy(net.features)
            print(base_net)
            base_net[split_point][0] = nn.Conv2d(in_channels, w, kernel_size=1, stride=1, padding=0, bias=False)
            torch.nn.init.xavier_uniform_(base_net[split_point][0].weight)
            
            base_net[split_point][1] = nn.BatchNorm2d(w)
            base_net[split_point].use_res_connect = False

            out_channels = base_net[split_point + 1].conv[0][0].out_channels
            base_net[split_point + 1].conv[0][0] = torch.nn.Conv2d(w, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            torch.nn.init.xavier_uniform_(base_net[split_point + 1].conv[0][0].weight)
            
            base_net[split_point + 1].use_res_connect = False
            base_net = base_net[split_point + 1:]
            self.heads[str(w)] = base_net

            self.linears[str(w)] = copy.deepcopy(net.classifier)

    def forward(self, x, width=None,quant_bits=0):
        if width is None:
            width = self.default_width
            if self.use_random_connect:
                width = np.random.choice(self.widths)
        print(width)
        out = self.shared_layers(x)
        out = out[:, 0:width, :, :]
        for _ in range(out.shape[1]):
            _max=torch.max(out[:,_])
            _min=torch.min(out[:,_])
            z_d=_max-_min
            if z_d<1e-6:
                z_d=1
            out[:,_]=_min+torch.round((out[:,_]-_min)*quant_bits/(z_d))*((z_d)/quant_bits)
			
			
        # x3=out.byte().cpu().numpy()
        # x3=numbits.pack(x3, nbits=qqq)
        # x3=pickle.dumps(x3)
        # print(sys.getsizeof(x3))
        # x3=pickle.loads(x3)
        # x3=np.array(x3)
        # x3=numbits.unpack(x3,nbits=qqq)
        # x4=torch.tensor(x3).float().cuda().reshape(out.shape)
        out = self.heads[str(width)](out)
        out = out.mean([2, 3])
        out = self.linears[str(width)](out)
        return out


def test():
    net = CLIOMobileNetV2()
    x = torch.randn(2,3,224,224)
    y = net(x)

# test()

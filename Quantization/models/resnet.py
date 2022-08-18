'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math
from models.quant_layer_s import *


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def Quantconv3x3(in_planes, out_planes, args, stride=1, bias=False):
    " 3x3 quantized convolution with padding "
    return QuantConv2d(in_planes, out_planes, kernel_size=3, args=args, stride=stride, padding=1, bias=bias)

def Quantconv1x1(in_planes, out_planes, args, stride=1, bias=False):
    return QuantConv2d(in_planes, out_planes, kernel_size=1, args=args, stride=stride, padding=0, bias=bias)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, args, stride=1, downsample=None, float=False):
        super(BasicBlock, self).__init__()
      
        if float:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv1 = Quantconv3x3(inplanes, planes, args, stride)
            self.conv2 = Quantconv3x3(planes, planes, args)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, args, stride=1, downsample=None, float=False):
        super(Bottleneck, self).__init__()
        if float:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        else:
            self.conv1 = Quantconv1x1(inplanes, planes, args, bias=False)
            self.conv2 = Quantconv3x3(planes, planes, args, stride, bias=False)
            self.conv3 = Quantconv1x1(planes, planes*4, args, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, args, num_classes=10, float=False):

        super(ResNet_Cifar, self).__init__()
        
        self.inplanes = 16
        self.args = args
        num_classes = args.num_classes
        self.conv1 = first_conv(3, 16, kernel_size=3, stride=1, padding=1, bias=False, args=args)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], args, float=float)
        self.layer2 = self._make_layer(block, 32, layers[1], args, stride=2, float=float)
        self.layer3 = self._make_layer(block, 64, layers[2], args, stride=2, float=float)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = last_fc(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, args, stride=1, float=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:   
            downsample = nn.Sequential(
                QuantConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, args=args)
                if float is False else nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                                 stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
            
        layers = []        
        layers.append(block(self.inplanes, planes, args, stride, downsample, float=float))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, args, float=float))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)

        return x

    def show_params(self):
        wgt_alpha_list = []
        act_alpha_list = []
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                w_value, a_value = m.show_params()
                wgt_alpha_list.append(w_value)
                act_alpha_list.append(a_value)
                
        print("wgt_alpha:", wgt_alpha_list)
        #print("\n")
        #print("act_alpha:", act_alpha_list)
        
def resnet20_cifar(args, **kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], args, **kwargs)
    return model

def resnet32_cifar(args, **kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], args, **kwargs)
    return model

def resnet44_cifar(args, **kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], args, **kwargs)
    return model


def resnet56_cifar(args, **kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], args, **kwargs)
    return model


def resnet110_cifar(args, **kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], args, **kwargs)
    return model


def resnet1202_cifar(args, **kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], args, **kwargs)
    return model


def resnet164_cifar(args, **kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], args, **kwargs)
    return model


#def resnet1001_cifar(**kwargs):
#    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
#    return model


if __name__ == '__main__':
    pass
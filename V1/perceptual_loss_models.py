from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import numpy as np


class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()

        self.down_conv_1 = DownConvBlock(3, 32, 9, 1, 4)
        self.down_conv_2 = DownConvBlock(32, 64, 4, 2, 1)
        self.down_conv_3 = DownConvBlock(64, 128, 4, 2, 1)

        self.res_1 = ResidualBlock(128)
        self.res_2 = ResidualBlock(128)
        self.res_3 = ResidualBlock(128)
        self.res_4 = ResidualBlock(128)
        self.res_5 = ResidualBlock(128)

        self.up_conv_1 = UpConvBlock(128, 64)
        self.up_conv_2 = UpConvBlock(64, 32)
        self.to_rgb = DownConvBlock(32, 3, 9, 1, 4, act='tanh')

    def forward(self, x):
        x = self.down_conv_1(x)
        x = self.down_conv_2(x)
        x = self.down_conv_3(x)

        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)

        x = self.up_conv_1(x)
        x = self.up_conv_2(x)
        return self.to_rgb(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3)
        self.bn1 = nn.InstanceNorm2d(ch, affine=True)
        self.conv2 = nn.Conv2d(ch, ch, 3)
        self.bn2 = nn.InstanceNorm2d(ch, affine=True)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        res = self.bn1(self.conv1(self.pad(x)))
        res = F.relu(res)
        res = self.bn2(self.conv2(self.pad(res)))
        return res + x


class DownConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, f_size, stride, padding, act='relu'):
        super(DownConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, f_size, stride=stride)
        self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        self.activation = nn.ReLU() if act == 'relu' else nn.Tanh()
        #self.activation = nn.ReLU()
        self.pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        return self.activation(self.bn(self.conv(self.pad(x))))


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConvBlock, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(in_ch, out_ch, 3)
        self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        return F.relu(self.bn(self.conv(self.pad(self.upsample(x)))))


class LossModel(nn.Module):
    def __init__(self, loss_layers_idx, loss_layer_names):
        super(LossModel, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        self.features = nn.ModuleList(features).eval()
        self.loss_layers_idx = loss_layers_idx
        self.loss_layers_names = loss_layer_names
        self.layer_weights = np.array([0.5, 0.6, 0.9, 1])
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.loss_layers_idx:
                results.append(x)
        results = [a * b for (a, b) in zip(results, self.layer_weights)]
        vgg_outputs = namedtuple("VggOutputs", self.loss_layers_names)
        return vgg_outputs(*results)

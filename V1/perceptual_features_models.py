from collections import namedtuple
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.features = nn.ReLU()
        self.pad = nn.ReflectionPad2d(1)
        self.clasifier = nn.Linear(64*3*3, 10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(self.pad(x))))
        x = self.pool2(self.relu(self.conv2(self.pad(x))))
        x = self.pool3(self.relu(self.conv3(self.pad(x))))
        x = x.view(-1, 64*3*3)
        x = self.features(x)
        return self.clasifier(x)


class FeatureExtractorModel(nn.Module):
    def __init__(self):
        super(FeatureExtractorModel, self).__init__()
        self.mnist_classifier = MnistClassifier()
        self.mnist_classifier.load_state_dict(torch.load('./mnist_classifier.pth'))
        self.mnist_classifier.eval()
        self.features_output = None
        self.pool_1_out = None
        self.pool_2_out = None

        def feat_hook(module, input_, output):
            nonlocal self
            self.features_output = output

        def pool1_hook(module, input_, output):
            nonlocal self
            self.pool_1_out = output.view(-1, 16*14*14)

        def pool2_hook(module, input_, output):
            nonlocal self
            self.pool_2_out = output.view(-1, 32*7*7)

        self.mnist_classifier.features.register_forward_hook(feat_hook)
        self.mnist_classifier.pool1.register_forward_hook(pool1_hook)
        self.mnist_classifier.pool2.register_forward_hook(pool2_hook)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        _ = self.mnist_classifier(x)
        return self.pool_1_out, self.pool_2_out, self.features_output


class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        self.dense_1 = nn.Linear(256, 128*7*7)
        #self.dense_2 = nn.Linear(256, 256*7*7)
        self.deconv_1 = UpConBlock(128, 256)
        self.deconv_2 = UpConBlock(256, 128)
        self.conv_1 = ConBlock(128, 64)
        self.to_rgb = nn.Conv2d(64, 1, 3)
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dense_1(x)
        #x = self.dense_2(x)
        x = x.view(-1, 128, 7, 7)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        #x = self.deconv_3(x)
        # x = self.relu(self.conv_1(x))
       # x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_1(x))
        x = self.to_rgb(self.pad(x))
        x = self.tanh(x)
        return x


class GeneratorModelCifar(nn.Module):
    def __init__(self):
        super(GeneratorModelCifar, self).__init__()
        self.dense_1 = nn.Linear(100, 4*4*512)
        #self.dense_2 = nn.Linear(256, 256*7*7)
        self.deconv_1 = DeconBlock(512, 256)  # 8x8
        self.deconv_1.apply(weights_init)
        self.deconv_2 = DeconBlock(256, 128)  # 16x16
        self.deconv_2.apply(weights_init)
        self.deconv_3 = DeconBlock(128, 64)  # 32x32
        self.deconv_3.apply(weights_init)
        self.conv_1 = ConBlock(64, 64)
        self.conv_1.apply(weights_init)
        self.conv_2 = ConBlock(64, 64)
        self.conv_2.apply(weights_init)
        self.to_rgb = nn.Conv2d(64, 3, 3)
        self.to_rgb.apply(weights_init)
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU()
        self.tanh = nn.Sigmoid()

    def forward(self, x):
        x = self.dense_1(x)
        x = x.view(-1, 512, 4, 4)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.tanh(self.to_rgb(self.pad(x)))
        return x


class UpConBlock(nn.Module):
    def __init__(self, in_c, out_c, ):
        super(UpConBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3)
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU()
        self.bn = nn.InstanceNorm2d(out_c)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(self.up(x)))))
        #return self.relu(self.conv(self.pad(self.up(x))))


class DeconBlock(nn.Module):
    def __init__(self, in_c, out_c,):
        super(DeconBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.InstanceNorm2d(out_c)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        #return self.relu(self.conv(x))


class ConBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_c, out_c, 3)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(self.pad(x)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

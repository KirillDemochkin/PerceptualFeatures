from collections import namedtuple

import torch
import torch.nn.functional as F
from torchvision import models

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg19_bn(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2.view(h_relu1_2.size(0), -1), h_relu2_2.view(h_relu2_2.size(0), -1), h_relu3_3.view(h_relu3_3.size(0), -1), h_relu4_3.view(h_relu4_3.size(0), -1))
        return out


class Vgg19Full(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19Full, self).__init__()
        vgg_pretrained_features = models.vgg19_bn(pretrained=True).features
        print(vgg_pretrained_features)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.slice8 = torch.nn.Sequential()
        self.slice9 = torch.nn.Sequential()
        self.slice10 = torch.nn.Sequential()
        self.slice11 = torch.nn.Sequential()
        self.slice12 = torch.nn.Sequential()
        self.slice13 = torch.nn.Sequential()
        self.slice14 = torch.nn.Sequential()
        self.slice15 = torch.nn.Sequential()
        self.slice16 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 6):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 10):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 13):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(13, 17):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(17, 20):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 23):
            self.slice7.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 26):
            self.slice8.add_module(str(x), vgg_pretrained_features[x])
        for x in range(26, 30):
            self.slice9.add_module(str(x), vgg_pretrained_features[x])
        for x in range(30, 33):
            self.slice10.add_module(str(x), vgg_pretrained_features[x])
        for x in range(33, 36):
            self.slice11.add_module(str(x), vgg_pretrained_features[x])
        for x in range(36, 39):
            self.slice12.add_module(str(x), vgg_pretrained_features[x])
        for x in range(39, 43):
            self.slice13.add_module(str(x), vgg_pretrained_features[x])
        for x in range(43, 46):
            self.slice14.add_module(str(x), vgg_pretrained_features[x])
        for x in range(46, 49):
            self.slice15.add_module(str(x), vgg_pretrained_features[x])
        for x in range(49, 52):
            self.slice16.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        h_relu6 = self.slice6(h_relu5)
        h_relu7 = self.slice7(h_relu6)
        h_relu8 = self.slice8(h_relu7)
        h_relu9 = self.slice9(h_relu8)
        h_relu10 = self.slice10(h_relu9)
        h_relu11 = self.slice11(h_relu10)
        h_relu12 = self.slice12(h_relu11)
        h_relu13 = self.slice13(h_relu12)
        h_relu14 = self.slice14(h_relu13)
        h_relu15 = self.slice15(h_relu14)
        h_relu16 = self.slice16(h_relu15)
        res = (h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7, h_relu8, h_relu9, h_relu10, h_relu11, h_relu12, h_relu13, h_relu14, h_relu15, h_relu16)
        #return [F.log_softmax(r.view(r.size(0), -1), dim=1) for r in res]
        return [r.view(r.size(0), -1) for r in res]
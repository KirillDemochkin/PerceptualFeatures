from functools import partial

import torch
import torch.nn as nn
from torchvision import models

class Resnet18Full(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet18Full, self).__init__()
        self.resnet_pretrained_features = models.resnet18(pretrained=True)
        print(self.resnet_pretrained_features)
        self.activations = []
        self.resnet_pretrained_features.relu.register_forward_hook(partial(self.save_activation))
        self.resnet_pretrained_features.layer1[0].relu.register_forward_hook(partial(self.save_activation))
        self.resnet_pretrained_features.layer1[1].relu.register_forward_hook(partial(self.save_activation))
        self.resnet_pretrained_features.layer2[0].relu.register_forward_hook(partial(self.save_activation))
        self.resnet_pretrained_features.layer2[1].relu.register_forward_hook(partial(self.save_activation))
        self.resnet_pretrained_features.layer3[0].relu.register_forward_hook(partial(self.save_activation))
        self.resnet_pretrained_features.layer3[1].relu.register_forward_hook(partial(self.save_activation))
        self.resnet_pretrained_features.layer4[0].relu.register_forward_hook(partial(self.save_activation))
        self.resnet_pretrained_features.layer4[1].relu.register_forward_hook(partial(self.save_activation))
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def save_activation(self, mod, inp, out):
        self.activations.append(out)

    def forward(self, x):
        self.activations = []
        _ = self.resnet_pretrained_features(x)
        return [r.view(r.size(0), -1) for r in self.activations]

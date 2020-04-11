import torch.nn as nn
from _collections import OrderedDict

class DCGAN(nn.Module):
    def __init__(self, latent_dim_size):
        super(DCGAN, self).__init__()
        self.latent_size = latent_dim_size
        self.proj = nn.ConvTranspose2d(self.latent_size, 64*4, 4, 1, 0, bias=False)
        self.proj_bn = nn.BatchNorm2d(64*4)
        self.proj_relu = nn.ReLU()
        self.proj.apply(weights_init)
        self.proj_bn.apply(weights_init)
        self.deconv1 = DeconBlock(64*4, 64*2)
        self.deconv1.apply(weights_init)
        self.deconv2 = DeconBlock(64*2, 64)
        self.deconv2.apply(weights_init)
        self.conv1 = ConBlock(64, 64)
        self.conv1.apply(weights_init)
        self.conv2 = ConBlock(64, 64)
        self.conv2.apply(weights_init)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        self.deconv3.apply(weights_init)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.proj_relu(self.proj_bn(self.proj(x.view(-1, self.latent_size, 1, 1))))
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.tanh(self.deconv3(x))
        return x


class UpConBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpConBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3)
        self.pad = nn.ReflectionPad2d(1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(self.up(x)))))


class DeconBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DeconBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConBlock, self).__init__()
        #self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        #return self.relu(self.bn(self.conv(self.pad(x))))
        return self.relu(self.bn(self.conv(x)))


class ResGenerator(nn.Module):
    def __init__(self, latent_dim_size):
        super(ResGenerator, self).__init__()
        num_layers = 4
        self.latent_dim_size = latent_dim_size
        filter_size_per_layer = [64] * num_layers
        for i in range(num_layers -1, -1, -1):
            if i == num_layers -1:
                filter_size_per_layer[i] = 64
            else:
                filter_size_per_layer[i] = filter_size_per_layer[i+1]*2
        first_l = nn.ConvTranspose2d(latent_dim_size, filter_size_per_layer[0], 4, 1, 0, bias=False)
        nn.init.xavier_uniform_(first_l.weight.data, 1.)
        last_l = nn.Conv2d(filter_size_per_layer[-1], 3, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(last_l.weight.data, 1.)

        nn_layers = OrderedDict()
        nn_layers["first_conv"] = first_l

        layer_number = 1
        for i in range(3):
            nn_layers["resblock_%d" % i] = ResBlock(filter_size_per_layer[layer_number-1], filter_size_per_layer[layer_number], stride=2)
            layer_number += 1
        nn_layers["batch_norm"] = nn.BatchNorm2d(filter_size_per_layer[-1])
        nn_layers["relu"] = nn.ReLU()
        nn_layers["last_conv"] = last_l
        nn_layers["tanh"] = nn.Tanh()
        self.net = nn.Sequential(nn_layers)

    def forward(self, x):
        return self.net(x.view(-1, self.latent_dim_size, 1, 1))


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, padding=1)
        self.conv_bypass = nn.Conv2d(in_c, out_c, 1, 1, padding=0)

        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv_bypass.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            self.conv2
        )

        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(self.conv_bypass, nn.UpsamplingNearest2d(scale_factor=2))

    def forward(self, x):
        return self.model(x) + self.bypass(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
        #nn.init.normal_(m.weight.data, 0.0, 0.02)
        #nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
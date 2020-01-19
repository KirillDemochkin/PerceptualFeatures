import torch.nn as nn


class DCGAN(nn.Module):
    def __init__(self, latent_dim_size):
        super(DCGAN, self).__init__()
        self.latent_size = latent_dim_size
        self.proj = nn.ConvTranspose2d(self.latent_size, 512, 4, 1, 0, bias=False)
        self.proj_bn = nn.BatchNorm2d(512)
        self.proj_relu = nn.ReLU()
        self.proj.apply(weights_init)
        self.proj_bn.apply(weights_init)
        self.deconv1 = DeconBlock(512, 256)
        self.deconv1.apply(weights_init)
        self.deconv2 = DeconBlock(256, 128)
        self.deconv2.apply(weights_init)
        self.deconv3 = DeconBlock(128, 64)
        self.deconv3.apply(weights_init)
        self.conv1 = ConBlock(64, 64)
        self.conv1.apply(weights_init)
        self.conv2 = ConBlock(64, 64)
        self.conv2.apply(weights_init)
        self.to_rgb_pad = nn.ReflectionPad2d(1)
        self.to_rgb = nn.Conv2d(64, 3, 3)
        self.to_rgb.apply(weights_init)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.proj_relu(self.proj_bn(self.proj(x.view(-1, self.latent_size, 1, 1))))
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.tanh(self.to_rgb(self.to_rgb_pad(x)))
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
        self.conv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_c, out_c, 3)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x))))


class ResGenerator(nn.Module):
    def __init__(self, latent_dim_size):
        super(ResGenerator, self).__init__()

    def forward(self, x):
        pass


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

    def forwards(self):
        pass


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

import torch.nn as nn


class DCGAN(nn.Module):
    def __init__(self, latent_dim_size):
        super(DCGAN, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim_size, latent_dim_size * 4, 4, 1, 0, bias=False)
        self.deconv1.apply(weights_init)
        self.first_bn = nn.InstanceNorm2d(latent_dim_size * 8)
        self.first_relu = nn.ReLU()

        self.deconv2 = DeconBlock(latent_dim_size * 4, latent_dim_size * 2)
        self.deconv2.apply(weights_init)
        self.deconv3 = DeconBlock(latent_dim_size * 2, latent_dim_size)
        self.deconv3.apply(weights_init)

        self.conv1 = ConBlock(latent_dim_size, latent_dim_size)
        self.conv1.apply(weights_init)
        self.conv2 = ConBlock(latent_dim_size, latent_dim_size)
        self.conv2.apply(weights_init)

        self.deconv4 = nn.ConvTranspose2d(latent_dim_size, 3, 4, 2, 1, bias=False)
        self.deconv4.apply(weights_init)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.first_relu(self.first_bn(self.deconv1))
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.deconv4(x)
        x = self.tanh(x)
        return x


class UpConBlock(nn.Module):
    def __init__(self, in_c, out_c, ):
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
        self.conv = nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_c, out_c, 3, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(self.pad(x)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.iinit.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

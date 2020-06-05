import functools

import torch
import torch.nn as nn

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(0.2, True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.spectral_norm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), eps=1e-8),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias), eps=1e-8),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), eps=1e-8), norm_layer(dim), nn.LeakyReLU(0.2, True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.utils.spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), eps=1e-8), norm_layer(dim)]

        return nn.Sequential(*conv_block)



    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Discriminator_block(nn.Module):
    def __init__(self, in_filters, out_filters, normalization=True):
        super(Discriminator_block, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), eps=1e-6)
        self.norm = nn.InstanceNorm2d(out_filters)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.normalization = normalization

    def forward(self, x):
        x = self.conv(x)
        if self.normalization:
            x = self.norm(x)
        x = self.leaky(x)
        return x


class MultiscaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.avg_pool = torch.nn.AvgPool2d(kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)
        self.subnetD1 = PatchDiscriminator()
        self.subnetD2 = PatchDiscriminator()
        #self.subnetD3 = GauGANDiscriminator(in_channels)

    def forward(self, x):
        outs = []
        preds = []
        out, pred = self.subnetD1(x)
        x = self.avg_pool(x)
        outs.append(out)
        preds.append(pred)
        out, pred = self.subnetD2(x)
        #x = self.avg_pool(x)
        #mask = self.avg_pool(mask)
        outs.append(out)
        preds.append(pred)
        #out, pred = self.subnetD3(x, mask)
        #outs.append(out)
        #preds.append(pred)
        return outs, preds


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.block_1 = Discriminator_block(3, 64,  normalization=False)
        self.block_2 = Discriminator_block(64, 128)
        self.block_3 = Discriminator_block(128, 256)
        self.block_4 = Discriminator_block(256, 512)
        self.out_conv = nn.Conv2d(512, 1, kernel_size=4, padding=1)

    def forward(self, x):
        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)
        preds = self.out_conv(x4)
        feats = (x1, x2, x3, x4)
        return preds, torch.cat([r.view(r.size(0), -1) for r in feats], dim=1)

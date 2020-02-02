# Third party
import pathlib
import torch
from torch import nn
from torchvision import models
import sys

# This project
from networks import utils



############################################################
# Feat. extractor returns features of a chosen pre-trained #
# network calculated for the input images                  #
#                                                          #
# Different pre-trained networks and layers can be used    #
############################################################

class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--ext_full_net_names', type=str, default='vgg19_imagenet_caffe, vgg16_face_caffe')
        parser.add('--ext_net_layers', type=str, default='1,6,11,20,29; 1,6,11,18,25', help='1,6,11,18,25 for VGGFace')
        parser.add('--ext_pooling', type=str, default='avgpool', help='maxpool|avgpool')
        parser.add('--ext_prefixes', type=str, default='real, fake', help='real|fake')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        self.prefixes = parse_str_to_list(args.ext_prefixes, ',')
        weights_dir = pathlib.Path(args.project_dir) / 'pretrained_weights' / 'extractor'

        # Architectures for the supported networks 
        networks = {
            'vgg16': models.vgg16,
            'vgg19': models.vgg19}

        # Build a list of used networks
        self.nets = nn.ModuleList()
        self.full_net_names = parse_str_to_list(args.ext_full_net_names, sep=',')

        for full_net_name in self.full_net_names:
            net_name, dataset_name, framework_name = full_net_name.split('_')
            if dataset_name == 'imagenet' and framework_name == 'pytorch':
                self.nets.append(networks[net_name](pretrained=True))
                mean = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None] * 2 - 1
                std  = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None] * 2
            elif framework_name == 'caffe':
                self.nets.append(networks[net_name]())
                self.nets[-1].load_state_dict(torch.load(weights_dir / f'{full_net_name}.pth'))
                self.nets[-1] = self.nets[-1]
                mean = torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] / 127.5 - 1
                std  = torch.FloatTensor([     1.,      1.,      1.])[None, :, None, None] / 127.5
            # Register means and stds as buffers
            self.register_buffer(f'{full_net_name}_mean', mean)
            self.register_buffer(f'{full_net_name}_std', std)

        # Perform the slicing according to the required layers
        for n, (net, block_idx) in enumerate(zip(self.nets, parse_str_to_list(args.ext_net_layers, sep=';'))):
            net_blocks = nn.ModuleList()

            # Parse indices of slices
            block_idx = parse_str_to_list(block_idx, sep=',')
            for i, idx in enumerate(block_idx):
                if idx.isdigit():
                    block_idx[i] = int(idx)

            # Slice conv blocks
            layers = []
            for i, layer in enumerate(net.features):
                if layer.__class__.__name__ == 'MaxPool2d' and args.ext_pooling == 'avgpool':
                    layer = nn.AvgPool2d(2)
                layers.append(layer)
                if i in block_idx:
                    net_blocks.append(nn.Sequential(*layers))
                    layers = []

            # Add layers for prediction of the scores (if needed)
            if block_idx[-1] == 'fc':
                layers.extend([
                    nn.AdaptiveAvgPool2d(7),
                    utils.Flatten(1)])
                for layer in net.classifier:
                    layers.append(layer)
                net_blocks.append(nn.Sequential(*layers))

            # Store sliced net
            self.nets[n] = net_blocks

    def forward(self, inputs):
        data_dict, aliases_to_train = inputs
        
        for prefix in self.prefixes:
            # Extract inputs
            imgs = data_dict[f'{prefix}_imgs']

            # Prepare inputs
            b, t, c, h, w = imgs.shape
            imgs = imgs.view(-1, c, h, w)

            # Calculate features
            feats = []
            for net, full_net_name in zip(self.nets, self.full_net_names):
                # Preprocess input image
                mean = getattr(self, f'{full_net_name}_mean')
                std = getattr(self, f'{full_net_name}_std')
                feats.append([(imgs - mean) / std])

                # Forward pass through blocks
                for block in net:
                    if prefix == 'real':
                        with torch.no_grad():
                            feats[-1].append(block(feats[-1][-1]))
                    else:
                        feats[-1].append(block(feats[-1][-1]))

                # Remove input image
                feats[-1].pop(0)

            # Store outputs
            data_dict[f'{prefix}_feats_ext'] = feats

        return data_dict


def parse_str_to_list(str, sep=','):
    return [s.replace(' ', '') for s in str.split(sep)]
import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from tqdm import tqdm

import utils
from transformer_net import TransformerNet
from vgg import Vgg16


def train():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder("./flickr30k_images", transform)
    train_loader = DataLoader(train_dataset, batch_size=4)

    transformer = TransformerNet()
    optimizer = Adam(transformer.parameters(), 0.0001)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image('./style_image.jpg', size=256)
    style = style_transform(style)
    style = style.repeat(4, 1, 1, 1)

    vgg.eval()
    with torch.no_grad():
        features_style = vgg(utils.normalize_batch(style))
        gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(2):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in tqdm(enumerate(train_loader)):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = 1e5 * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= 1e10

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id) % 20 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)
                stylize(transformer, './test_res/%d_%d.jpg' % (e, batch_id))


def stylize(style_model, name):
    content_image = utils.load_image('./test_img.jpg')
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)

    style_model.eval()
    with torch.no_grad():
        output = style_model(content_image)
    style_model.train()
    utils.save_image(name, output[0])


def main():
   train()

if __name__ == "__main__":
    main()
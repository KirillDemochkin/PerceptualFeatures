import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from perceptual_loss_models import TransformNet, LossModel

from PIL import Image

transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(256),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

to_image = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) * 127.5),
])


def preprocess_vgg(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.add(1).mul(127.5)
    batch = batch.div(255.0)
    return (batch - mean) / std


BATCH_SIZE = 4

style_image = transform(Image.open("./tsunami.jpg"))
test_image = transform_test(Image.open("./test_img.jpg")).unsqueeze(0)


train_ds = torchvision.datasets.ImageFolder(root="./flickr30k_images", transform=transform)
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
transform_net = TransformNet()
loss_model = LossModel({3, 8, 15, 22}, ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
loss_model.eval()
criterion = nn.MSELoss()


def gram_matrix(output):
    n, c, h, w = output.size()
    features = output.view(n, c, h * w)
    G = features.bmm(features.transpose(1, 2))
    return G / (c * h * w)

style_image_batch = style_image.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1)
target_gram_matrices = [gram_matrix(feats) for feats in loss_model(preprocess_vgg(style_image_batch))]

def tv_loss(output):
    return torch.sum(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) + \
           torch.sum(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :]))


def perceptual_loss(output, target):
    output_activations = loss_model(preprocess_vgg(output))
    target_activations = loss_model(preprocess_vgg(target))
    reconstruction_loss = criterion(output_activations.relu2_2, target_activations.relu2_2)
    output_gram_matrices = [gram_matrix(feats) for feats in output_activations]

    style_loss = 0
    for o_g, t_g in zip(output_gram_matrices, target_gram_matrices):
        style_loss += criterion(o_g, t_g[:output.shape[0], :, :])
    style_loss *= 1e5
    reconstruction_loss *= 1e2
    return reconstruction_loss.item(), style_loss.item(),  reconstruction_loss + style_loss
    #return reconstruction_loss.item(), style_loss.item(),  reconstruction_loss + style_loss + 1e-8 * tv_loss(output) + 0.2 * criterion(output, target)


def test_stylization(name):
    transform_net.eval()
    with torch.no_grad():
        stylized_test = transform_net(test_image)
    transform_net.train()
    im = to_image(stylized_test.squeeze(0))
    im = im.clamp(0, 255).numpy()
    im = im.transpose(1, 2, 0).astype("uint8")
    im = Image.fromarray(im)
    im.save('./test_res/%s.jpg' % name)


optimizer = torch.optim.Adam(transform_net.parameters(), lr=0.000001)
transform_net.train()

for epoch in range(2):
    total_epoch_style_loss = 0
    total_epoch_content_loss = 0
    total_epoch_perceptual_loss = 0
    av_epoch_style_loss = 0
    av_epoch_content_loss = 0
    av_epoch_perceptual_loss = 0

    num_processed = 0
    print()
    print('~~~Epoch %d ~~~' % epoch)
    for i, batch in tqdm(enumerate(trainloader, 0)):
        x, _ = batch
        optimizer.zero_grad()
        stylized = transform_net(x)

        cont_batch_loss, style_batch_loss, loss = perceptual_loss(stylized, x)
        loss.backward()
        optimizer.step()

        num_processed += x.shape[0]
        #print(loss.item())
        total_epoch_content_loss += cont_batch_loss * num_processed
        total_epoch_style_loss += style_batch_loss * num_processed
        total_epoch_perceptual_loss += loss.item() * num_processed
        av_epoch_content_loss = cont_batch_loss
        av_epoch_style_loss = style_batch_loss
        av_epoch_perceptual_loss = loss.item()

        if i % 20 == 0:
            print('average style loss %.4f | average content loss %.4f | average perceptual loss %.4f' %
                  (av_epoch_style_loss, av_epoch_content_loss, av_epoch_perceptual_loss))
            test_stylization("%d_%d" % (epoch, i))
    print()
    test_stylization(epoch)
    print('EPOCH %d SUMMARY:\ntotal style loss %.4f\ntotal content loss %.4f\naverage perceptual loss %.4f' %
          (epoch, total_epoch_style_loss, total_epoch_style_loss, total_epoch_perceptual_loss))


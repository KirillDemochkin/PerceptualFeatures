from multiprocessing import freeze_support

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from vgg import Vgg16
from torch.nn.utils import clip_grad_norm_
from perceptual_features_models import GeneratorModelCifar

import os

transform = transforms.Compose(
    [transforms.ToTensor()]  #maybe noralize to [-1, 1] ?
)

test_noise = torch.tensor(np.random.normal(0, 1, (32, 100))).float()


def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


def precompute_mean_and_cov(target_layers):
    bs = 64
    num_layers = len(target_layers)
    feature_extractor = Vgg16()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True)
    features = {i: [] for i in range(num_layers)}
    with torch.no_grad():
        for (ind, (x, _)) in tqdm(enumerate(trainloader, 0)):
            out = feature_extractor(normalize_batch(x))
            for i in range(num_layers):
                features[i].append(out[i].view(len(x), -1).numpy())
        for i in range(num_layers):
            features[i] = np.concatenate(features[i], axis=0)
            np.save('./feats/%s_m.npy' % target_layers[i], features[i].mean(0))
            np.save('./feats/%s_v.npy' % target_layers[i], (features[i]**2).mean(0) - features[i].mean(0)**2)
            del features[i]
            features[i] = None


def update_moving_averages(delta, mt, ut, t, b1=0.9, b2=0.999, eps=1e-8):
    mt1 = b1 * mt + (1 - b1) * delta
    mt1_hat = mt1 / (1 - b1**t)

    ut1 = b2 * ut + (1 - b2) * torch.pow(delta, 2)
    ut1_hat = ut1 / (1 - b2**t)

    return mt1, ut1, torch.div(mt1_hat, torch.add(torch.sqrt(ut1_hat), eps))


def compute_moments(features):
    ms = torch.mean(features, dim=0)
    vs = torch.sub(torch.mean(torch.pow(features, 2), dim=0), torch.pow(ms, 2))
    return ms, vs


def featstats(feats, name):
    print(name, np.max(feats), np.min(feats), np.count_nonzero(feats))


def train_generator_ma(mean_fs, var_fs, num_layers, layer_weights):
    feature_extractor = Vgg16()
    feature_extractor.eval()
    generator = GeneratorModelCifar()
    generator.train()
    mean_fs = [m.detach() for m in mean_fs]
    var_fs = [v.detach() for v in var_fs]
    bs = 64
    running_means_dt = []
    running_covs_dt = []
    with torch.no_grad():
        #g = generator(torch.tensor(np.random.normal(0, 1, (bs, 100))).float())
        #initial_preds = [fe.view(bs, -1) for fe in feature_extractor(g)] # try setting to zero
        for i in range(num_layers):
            #ms, vs = compute_moments(initial_preds[i])
            #running_means_dt.append(mean_fs[i] - ms)
            #running_covs_dt.append(var_fs[i] - vs)
            running_means_dt.append(torch.zeros_like(mean_fs[i]))
            running_covs_dt.append(torch.zeros_like(var_fs[i]))

    mt_means = [torch.zeros_like(running_means_dt[i]) for i in range(num_layers)]
    ut_means = [torch.zeros_like(running_means_dt[i]) for i in range(num_layers)]
    mt_covs = [torch.zeros_like(running_covs_dt[i]) for i in range(num_layers)]
    ut_covs = [torch.zeros_like(running_covs_dt[i]) for i in range(num_layers)]

    generator_optim = optim.Adam(params=generator.parameters(), lr=1e-4)
    #generator_optim = optim.SGD(params=generator.parameters(), lr=1e-5)
    for step in range(2000):
        loss = 0
        generator_optim.zero_grad()
        generated_features = [fe.view(bs, -1) for fe in feature_extractor(normalize_batch(generator(torch.tensor(np.random.normal(0, 1, (bs, 100))).float())))]
        batch_delta_m = []
        batch_delta_c = []
        for i in range(num_layers):
            ms, vs = compute_moments(generated_features[i])
            batch_delta_m.append(mean_fs[i]-ms)
            batch_delta_c.append(var_fs[i]-vs)

        batch_loss = torch.zeros((1,)).float()
        for i in range(num_layers):
            ms_loss = torch.abs(torch.matmul(running_means_dt[i], batch_delta_m[i]))
            vs_loss = torch.abs(torch.matmul(running_covs_dt[i], batch_delta_c[i]))
            batch_loss += torch.add(ms_loss, vs_loss)
        loss += batch_loss.item()
        batch_loss.backward()

        #total_norm = torch.zeros((1,)).float()
        #for p in generator.parameters():
            #param_norm = p.grad.data.norm(2)
            #total_norm += param_norm.item() ** 2
        #total_norm = total_norm ** (1. / 2)
        clip_grad_norm_(generator.parameters(), 5, norm_type=2)
        generator_optim.step()
        with torch.no_grad():
            for i in range(num_layers):
                mt_means[i], ut_means[i], adam_mean = update_moving_averages(running_means_dt[i] - batch_delta_m[i], mt_means[i], ut_means[i], step+1)
                mt_covs[i], ut_covs[i], adam_cov = update_moving_averages(running_covs_dt[i] - batch_delta_c[i], mt_covs[i], ut_covs[i], step+1)

                running_means_dt[i] -= 5*1e-5 * adam_mean
                running_covs_dt[i] -= 5*1e-5 * adam_cov

        if step % 5 == 0:
            print('batch %d: loss: %.4f' % (step, loss / (step+1)))
            visualize_sample(generator, "%d.jpg" % (step))
            torch.save(generator.state_dict(), './generator.pth')


def visualize_sample(generator, name):
    generator.eval()
    with torch.no_grad():
        pred = generator(test_noise)
        pred = pred * 255.0
        for i in range(len(pred)):

            im = pred[i].clamp(0, 255).squeeze().numpy()
            im = im.astype("uint8")
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im).resize((320, 320))
            im.save('./gen_res/%d/%s.jpg' % (i, name))
    generator.train()


def main():
    target_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
    layers_weights = [1/4.0, 1/4.0, 1/4.0, 1/4.0]
    if not os.path.exists('./feats/relu1_2_m.npy') or not os.path.exists('./feats/relu1_2_v.npy'):
        precompute_mean_and_cov(target_layers)
    else:
        means = ['./feats/%s_m.npy' % s for s in target_layers]
        variances = ['./feats/%s_v.npy' % s for s in target_layers]

        precomputed_means = [torch.tensor(np.load(l), dtype=torch.float32) for l in means]
        precomputed_variances = [torch.tensor(np.load(l), dtype=torch.float32) for l in variances]
        for i in range(len(target_layers)):
            featstats(precomputed_means[i].numpy(), target_layers[i])
            featstats(precomputed_variances[i].numpy(), target_layers[i])
        train_generator_ma(precomputed_means, precomputed_variances, len(target_layers), layers_weights)


if __name__ == '__main__':
    main()

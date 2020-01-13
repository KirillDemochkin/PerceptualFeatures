import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from perceptual_features_models import MnistClassifier, FeatureExtractorModel, GeneratorModel

import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

to_img = transforms.ToPILImage()

trainset = torchvision.datasets.mnist.MNIST(root='./mnist', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

test_noise = torch.tensor(np.random.normal(0, 0.1, (32, 256)), dtype=torch.float32)
LR = 1e-3

def train_mnist_classifier():
    mnist_classifier = MnistClassifier()
    mnist_optim = optim.Adam(params=mnist_classifier.parameters(), lr=0.00005)
    criterion = nn.CrossEntropyLoss()
    for ep in range(2):
        loss = 0
        for i, (x, y) in tqdm(enumerate(trainloader, 0)):
            preds = mnist_classifier(x)
            batch_loss = criterion(preds, y)
            batch_loss.backward()
            mnist_optim.step()

            loss = batch_loss.item() / len(x)

            if i % 100 == 0:
                print('batch %d: loss: %.4f' % (i, loss))
    torch.save(mnist_classifier.state_dict(), './mnist_classifier.pth')


def precompute_mean_and_cov():
    feature_extractor = FeatureExtractorModel()
    total = 0
    features1 = []
    features2 = []
    features3 = []
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(trainloader, 0)):
            total += len(x)
            (f1, f2, f3) = feature_extractor(x)
            features1.append(f1.numpy())
            features2.append(f2.numpy())
            features3.append(f3.numpy())

    features1 = np.concatenate(features1, axis=0)
    features2 = np.concatenate(features2, axis=0)
    features3 = np.concatenate(features3, axis=0)

    np.save('mean_1', features1.mean(0))
    np.save('var_1', np.var(features1, axis=0))

    np.save('mean_2', features2.mean(0))
    np.save('var_2', np.var(features2, axis=0))

    np.save('mean_3', features3.mean(0))
    np.save('var_3', np.var(features3, axis=0))

def train_generator_fml(mean_fs):
    feature_extractor = FeatureExtractorModel()
    feature_extractor.eval()
    generator = GeneratorModel()
    generator.train()
    criterion = nn.MSELoss()
    generator_optim = optim.Adam(params=generator.parameters(), lr=0.00005)
    for ep in range(2):
        loss = 0
        for i, (x, y) in tqdm(enumerate(trainloader, 0)):
            generator_optim.zero_grad()
            z = torch.tensor(np.random.normal(0, 1, (32, 256)), dtype=torch.float32)
            generated = generator(z)
            #target_features = feature_extractor(x)
            generated_features = feature_extractor(generated)
            batch_loss = criterion(generated_features, mean_fs)
            batch_loss.backward()
            generator_optim.step()

            loss = batch_loss.item() / len(x)

            if i % 100 == 0:
                print('batch %d: loss: %.4f' % (i, loss))
                visualize_sample(generator, "%d_%d.jpg" % (ep, i))
        torch.save(generator.state_dict(), './generator.pth')


def update_moving_averages(delta, mt, ut, t, b1=0.9, b2=0.999, eps=1e-8):
    mt1 = b1 * mt + (1-b1) * delta
    mt1_hat = mt1 / (1 - b1**t)

    ut1 = b2 * ut + (1 - b2) * torch.pow(delta, 2)
    ut1_hat = ut1 / (1 - b2**t)

    return mt1, ut1, torch.div(mt1_hat, torch.add(torch.sqrt(ut1_hat), eps))


def cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def train_generator_ma(mean_fs1, var_fs1, mean_fs2, var_fs2, mean_fs3, var_fs3):
    feature_extractor = FeatureExtractorModel()
    feature_extractor.eval()
    generator = GeneratorModel()
    generator.train()
    mean_fs1 = mean_fs1.detach()
    mean_fs2 = mean_fs2.detach()
    mean_fs3 = mean_fs3.detach()
    var_fs1 = var_fs1.detach()
    var_fs2 = var_fs2.detach()
    var_fs3 = var_fs3.detach()


    with torch.no_grad():
        g = generator(torch.tensor(np.random.normal(0, 0.1, (32, 256)), dtype=torch.float32))
        (initial_preds1, initial_preds2, initial_preds3) = feature_extractor(g)
        running_mean_dt1 = mean_fs1 - torch.mean(initial_preds1, dim=0)
        running_cov_dt1 = var_fs1 - torch.var(initial_preds1, dim=0)
        running_mean_dt2 = mean_fs2 - torch.mean(initial_preds2, dim=0)
        running_cov_dt2 = var_fs2 - torch.var(initial_preds2, dim=0)
        running_mean_dt3 = mean_fs3 - torch.mean(initial_preds3, dim=0)
        running_cov_dt3 = var_fs3 - torch.var(initial_preds3, dim=0)

    mt_mean1 = torch.zeros_like(running_mean_dt1)
    ut_mean1 = torch.zeros_like(running_mean_dt1)
    mt_cov1 = torch.zeros_like(running_cov_dt1)
    ut_cov1 = torch.zeros_like(running_cov_dt1)

    mt_mean2 = torch.zeros_like(running_mean_dt2)
    ut_mean2 = torch.zeros_like(running_mean_dt2)
    mt_cov2 = torch.zeros_like(running_cov_dt2)
    ut_cov2 = torch.zeros_like(running_cov_dt2)

    mt_mean3 = torch.zeros_like(running_mean_dt3)
    ut_mean3 = torch.zeros_like(running_mean_dt3)
    mt_cov3 = torch.zeros_like(running_cov_dt3)
    ut_cov3 = torch.zeros_like(running_cov_dt3)

    generator_optim = optim.Adam(params=generator.parameters(), lr=1e-5)
    for step in range(2000):
        loss = 0

        generator_optim.zero_grad()
        z = torch.tensor(np.random.normal(0, 0.1, (32, 256)), dtype=torch.float32)
        generated = generator(z)
        (generated_features1, generated_features2, generated_features3) = feature_extractor(generated)

        batch_delta_m1 = mean_fs1-torch.mean(generated_features1, dim=0)
        batch_delta_c1 = var_fs1-torch.var(generated_features1, dim=0)
        batch_delta_m2 = mean_fs2 - torch.mean(generated_features2, dim=0)
        batch_delta_c2 = var_fs2 - torch.var(generated_features2, dim=0)
        batch_delta_m3 = mean_fs3 - torch.mean(generated_features3, dim=0)
        batch_delta_c3 = var_fs3 - torch.var(generated_features3, dim=0)

        batch_loss = 0.2 * (torch.matmul(running_mean_dt1.unsqueeze(0), batch_delta_m1) + torch.matmul(running_cov_dt1.unsqueeze(0), batch_delta_c1)) + \
                     0.3 * (torch.matmul(running_mean_dt2.unsqueeze(0), batch_delta_m2) + torch.matmul(running_cov_dt2.unsqueeze(0), batch_delta_c2)) + \
                     0.5 * (torch.matmul(running_mean_dt3.unsqueeze(0), batch_delta_m3) + torch.matmul(running_cov_dt3.unsqueeze(0), batch_delta_c3))
        loss += batch_loss.item()
        batch_loss.backward()
        generator_optim.step()
        with torch.no_grad():
            mt_mean1, ut_mean1, adam_mean1 = update_moving_averages(running_mean_dt1 - batch_delta_m1, mt_mean1, ut_mean1, step+1)
            mt_cov1, ut_cov1, adam_cov1 = update_moving_averages(running_cov_dt1 - batch_delta_c1, mt_cov1, ut_cov1, step+1)
            running_mean_dt1 = running_mean_dt1 - LR * adam_mean1
            running_cov_dt1 = running_cov_dt1 - LR * adam_cov1

            mt_mean2, ut_mean2, adam_mean2 = update_moving_averages(running_mean_dt2 - batch_delta_m2, mt_mean2, ut_mean2, step + 1)
            mt_cov2, ut_cov2, adam_cov2 = update_moving_averages(running_cov_dt2 - batch_delta_c2, mt_cov2, ut_cov2, step + 1)
            running_mean_dt2 = running_mean_dt2 - LR * adam_mean2
            running_cov_dt2 = running_cov_dt2 - LR * adam_cov2

            mt_mean3, ut_mean3, adam_mean3 = update_moving_averages(running_mean_dt3 - batch_delta_m3, mt_mean3, ut_mean3, step + 1)
            mt_cov3, ut_cov3, adam_cov3= update_moving_averages(running_cov_dt3 - batch_delta_c3, mt_cov3, ut_cov3, step + 1)
            running_mean_dt3 = running_mean_dt3 - LR * adam_mean3
            running_cov_dt3 = running_cov_dt3 - LR * adam_cov3

        if step % 20 == 0:
            print('batch %d: loss: %.4f' % (step, loss / (step+1)))
            visualize_sample(generator, "%d.jpg" % (step))
            torch.save(generator.state_dict(), './generator.pth')


def visualize_sample(generator, name):
    generator.train()
    with torch.no_grad():
        pred = generator(test_noise)[0]
        pred = (pred + 1) * 127.5
        im = pred.clamp(0, 255).squeeze().numpy()
        im = im.astype("uint8")
        im = Image.fromarray(im, mode='L').resize((280, 280))
        im.save('./gen_res/%s.jpg' % name)

if not os.path.exists('./mnist_classifier.pth'):
    train_mnist_classifier()
else:
    if not os.path.exists('./mean_1.npy') or not os.path.exists('./var_1.npy'):
        precompute_mean_and_cov()
    else:
        precomputed_mean1 = np.load('./mean_1.npy')
        precomputed_cov1 = np.load('./var_1.npy')
        precomputed_mean2 = np.load('./mean_2.npy')
        precomputed_cov2 = np.load('./var_2.npy')
        precomputed_mean3 = np.load('./mean_3.npy')
        precomputed_cov3 = np.load('./var_3.npy')
        train_generator_ma(torch.tensor(precomputed_mean1, dtype=torch.float32), torch.tensor(precomputed_cov1, dtype=torch.float32),
                           torch.tensor(precomputed_mean2, dtype=torch.float32), torch.tensor(precomputed_cov2, dtype=torch.float32),
                           torch.tensor(precomputed_mean3, dtype=torch.float32), torch.tensor(precomputed_cov3, dtype=torch.float32))

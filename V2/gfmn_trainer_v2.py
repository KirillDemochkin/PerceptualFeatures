import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

import numpy as np
import os

from models.vgg import Vgg19Full
from models.generator_models import DCGAN, ResGenerator


BATCH_SIZE = 64
LATENT_DIM = 100
B1 = 0.8
LR_G = 5e-6
LR_MV_AVG = 1e-6
NUM_ITERATIONS = int(2e6)
SAVE_MODEL_ITERS = 500
SAMPLE_IMGS_ITERS = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

imageNetNormMean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
imageNetNormStd = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
imageNetNormMin = -imageNetNormMean / imageNetNormStd
imageNetNormMax = (1.0 - imageNetNormMean) / imageNetNormStd
imageNetNormRange = imageNetNormMax - imageNetNormMin

imageNetNormMin = torch.tensor(imageNetNormMin, dtype=torch.float32).view(1, 3, 1, 1).to(device)
imageNetNormRange = torch.tensor(imageNetNormRange, dtype=torch.float32).view(1, 3, 1, 1).to(device)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

test_noise = torch.empty(64, LATENT_DIM).normal_(mean=0, std=1).to(device)

print("Loading VGG")
vgg_pretrained = Vgg19Full().to(device).eval()
generator = ResGenerator(LATENT_DIM).to(device)
#generator = DCGAN(LATENT_DIM).to(device)

# normalize tensors
# count total features
total_features = 0
with torch.no_grad():
    empty_res = vgg_pretrained(torch.empty(4, 3, 32, 32).normal_(mean=0, std=1).to(device))
    print([er.shape[1] for er in empty_res])
for r in empty_res:
    total_features += r.shape[1]
print('Performing feature matching for %d features' % total_features)
# mean/var nets, losses, and optimizers
mean_net = nn.Linear(total_features, 1, bias=False).to(device)
var_net = nn.Linear(total_features, 1, bias=False).to(device)

criterionLossL2 = nn.MSELoss().to(device)

parametersG = set()
parametersG |= set(generator.parameters())
optimizerG = optim.Adam(parametersG, LR_G, betas=(B1, 0.999))
optimizerM = optim.Adam(mean_net.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerV = optim.Adam(var_net.parameters(), LR_MV_AVG, betas=(B1, 0.999))


# feature extraction
def save_models(suffix=""):
    # saving current best model
    torch.save(generator.state_dict(), './exported_models/generator%s.pth' % suffix)
    torch.save(mean_net.state_dict(), './exported_models/netMean%s.pth' % suffix)
    torch.save(var_net.state_dict(), './exported_models/netVar%s.pth' % suffix)


def sample_images():
    generator.eval()
    test_gen = generator(test_noise)
    vutils.save_image(test_gen.data[:64], './generated_samples/%d.png' % int((i + 1)/SAMPLE_IMGS_ITERS), normalize=True)
    generator.train()


def extract_features_from_batch(batch):
    feats = []
    vgg_out = vgg_pretrained(batch)
    for j in range(batch.size(0)):
        ft_sample = torch.cat([ft[j, :] for ft in vgg_out], dim=0)
        feats.append(ft_sample.view(1, -1))
    return torch.cat(feats, dim=0)


real_mean = torch.load('./data/mean.pt') if os.path.exists('./data/mean.pt') else None
real_sqr = None
real_var = torch.load('./data/var.pt') if os.path.exists('./data/var.pt') else None
num_processed = 0.0
first_time = True
if real_mean is None:
    with torch.no_grad():
        for i, data in tqdm(enumerate(trainloader, 1)):
            img_batch, _ = data
            img_batch = img_batch.to(device)
            extracted_batch = extract_features_from_batch(img_batch)

            if first_time:
                real_mean = torch.sum(extracted_batch, dim=0)
                real_sqr = torch.sum(extracted_batch ** 2, dim=0)
                first_time = False
            else:
                real_mean = torch.add(real_mean, torch.sum(extracted_batch, dim=0))
                real_sqr = torch.add(real_sqr, torch.sum(extracted_batch ** 2, dim=0))

            num_processed += img_batch.size(0)

        real_var = (real_sqr - (real_mean ** 2) / num_processed) / (
                num_processed - 1)
        print('normalizing by %.2f' % num_processed)
        real_mean = real_mean / num_processed

        torch.save(real_mean, './data/mean.pt')
        torch.save(real_var, './data/var.pt')

# training loop
avrg_g_var_net_loss = 0.0
avrg_g_mean_net_loss = 0.0
avrg_mean_net_loss = 0.0
avrg_var_net_loss = 0.0
avrg_g_total_loss = 0.0
os.sys.stdout.flush()
for i in tqdm(range(NUM_ITERATIONS)):
    generator.zero_grad()
    mean_net.zero_grad()
    var_net.zero_grad()
    vgg_pretrained.zero_grad()

    noise_batch = torch.empty(BATCH_SIZE, LATENT_DIM).normal_(mean=0, std=1).to(device)
    fake_imgs = generator(noise_batch)
    #fake_imgs = ((fake_imgs*0.5 + 0.5) - imageNetNormMean) / imageNetNormStd
    fake_imgs = (((fake_imgs + 1) * imageNetNormRange) / 2) + imageNetNormMin
    fake_features = extract_features_from_batch(fake_imgs)

    fake_mean = torch.mean(fake_features, 0)
    real_fake_difference_mean = real_mean.detach() - fake_mean.detach()
    mean_net_loss = criterionLossL2(mean_net.weight, real_fake_difference_mean.detach().view(1, -1))
    mean_net_loss.backward()
    avrg_mean_net_loss += mean_net_loss.item()
    clip_grad_norm_(mean_net.parameters(), 2, norm_type=2)
    optimizerM.step()

    fake_var = torch.var(fake_features, 0)
    real_fake_difference_var = real_var.detach() - fake_var.detach()
    var_net_loss = criterionLossL2(var_net.weight, real_fake_difference_var.detach().view(1, -1))
    var_net_loss.backward()
    avrg_var_net_loss += var_net_loss.item()
    clip_grad_norm_(var_net.parameters(), 2, norm_type=2)
    optimizerV.step()

    mean_diff_real = mean_net(real_mean.view(1, -1)).detach()
    mean_diff_fake = mean_net(fake_mean.view(1, -1))
    var_diff_real = var_net(real_var.view(1, -1)).detach()
    var_diff_fake = var_net(fake_var.view(1, -1))

    g_mean_net_loss = (mean_diff_real - mean_diff_fake)
    avrg_g_mean_net_loss += g_mean_net_loss.item()

    g_var_net_loss = (var_diff_real - var_diff_fake)
    avrg_g_var_net_loss += g_var_net_loss.item()

    generator_loss = g_mean_net_loss + g_var_net_loss
    generator_loss.backward()
    avrg_g_total_loss += generator_loss.item()
    clip_grad_norm_(generator.parameters(), 2, norm_type=2)
    optimizerG.step()

    # saving models/images
    if (i + 1) % SAMPLE_IMGS_ITERS == 0:
        print('Loss_G_total: %.6f Loss_Gz: %.6f Loss_GzVar: %.6f Loss_vMean: %.6f Loss_vVar: %.6f' %
              (avrg_g_total_loss / SAMPLE_IMGS_ITERS,
               avrg_g_mean_net_loss / SAMPLE_IMGS_ITERS, avrg_g_var_net_loss / SAMPLE_IMGS_ITERS,
               avrg_mean_net_loss / SAMPLE_IMGS_ITERS, avrg_var_net_loss / SAMPLE_IMGS_ITERS))
        os.sys.stdout.flush()
        with torch.no_grad():
            sample_images()

        avrg_g_var_net_loss = 0.0
        avrg_g_mean_net_loss = 0.0
        avrg_mean_net_loss = 0.0
        avrg_var_net_loss = 0.0
        avrg_g_total_loss = 0.0

    if (i + 1) % SAVE_MODEL_ITERS == 0:
        save_models("")

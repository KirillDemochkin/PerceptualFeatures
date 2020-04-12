import functools

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from tqdm import tqdm

import numpy as np
import os

from models.vgg import Vgg19Full
from models.cycle_gan_models import ResnetGenerator
from models.generator_models import weights_init

BATCH_SIZE = 64
B1 = 0.5
LR_G = 5e-5
LR_MV_AVG = 1e-5
NUM_ITERATIONS = int(2e6)
IMG_SIZE = 128
SAVE_MODEL_ITERS = 500
SAMPLE_IMGS_ITERS = 500
CYCLE_LAMBDA = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

imageNetNormMean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
imageNetNormStd = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
imageNetNormMin = -imageNetNormMean / imageNetNormStd
imageNetNormMax = (1.0 - imageNetNormMean) / imageNetNormStd
imageNetNormRange = imageNetNormMax - imageNetNormMin

imageNetNormMin = torch.tensor(imageNetNormMin, dtype=torch.float32).view(1, 3, 1, 1).to(device)
imageNetNormRange = torch.tensor(imageNetNormRange, dtype=torch.float32).view(1, 3, 1, 1).to(device)

transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset_horses = datasets.ImageFolder(root='./data/horse2zebra/horses', transform=transform)
trainloader_horses = torch.utils.data.DataLoader(trainset_horses, batch_size=16, shuffle=True, num_workers=0)

trainset_zebras = datasets.ImageFolder(root='./data/horse2zebra/zebras', transform=transform)
trainloader_zebras = torch.utils.data.DataLoader(trainset_zebras, batch_size=16, shuffle=True, num_workers=0)

print("Loading VGG")
vgg_pretrained = Vgg19Full().to(device).eval()

generator_horses = ResnetGenerator(input_nc=3,
                                   output_nc=3,
                                   ngf=64,
                                   norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
                                   use_dropout=False,
                                   n_blocks=6,
                                   padding_type='reflect').to(device)
generator_zebras = ResnetGenerator(input_nc=3,
                                   output_nc=3,
                                   ngf=64,
                                   norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False),
                                   use_dropout=False,
                                   n_blocks=6,
                                   padding_type='reflect').to(device)

generator_zebras.apply(weights_init)
generator_horses.apply(weights_init)

total_features = 0
with torch.no_grad():
    empty_res = vgg_pretrained(torch.empty(4, 3, IMG_SIZE, IMG_SIZE).normal_(mean=0, std=1).to(device))
    print([er.shape[1] for er in empty_res])
for r in empty_res:
    total_features += r.shape[1]
print('Performing feature matching for %d features' % total_features)
# mean/var nets, losses, and optimizers
mean_net_horses = nn.Linear(total_features, 1, bias=False).to(device)
var_net_horses = nn.Linear(total_features, 1, bias=False).to(device)

mean_net_zebras = nn.Linear(total_features, 1, bias=False).to(device)
var_net_zebras = nn.Linear(total_features, 1, bias=False).to(device)

criterionLossL2 = nn.MSELoss().to(device)
criterionCycleLoss = nn.L1Loss().to(device)

parametersG_horses = set()
parametersG_horses |= set(generator_horses.parameters())
optimizerG_horses = optim.Adam(parametersG_horses, LR_G, betas=(B1, 0.999))
optimizerM_horses = optim.Adam(mean_net_horses.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerV_horses = optim.Adam(var_net_horses.parameters(), LR_MV_AVG, betas=(B1, 0.999))

parametersG_zebras = set()
parametersG_zebras |= set(generator_zebras.parameters())
optimizerG_zebras = optim.Adam(parametersG_zebras, LR_G, betas=(B1, 0.999))
optimizerM_zebras = optim.Adam(mean_net_zebras.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerV_zebras = optim.Adam(var_net_zebras.parameters(), LR_MV_AVG, betas=(B1, 0.999))

def save_models(suffix=""):
    # saving current best model
    torch.save(generator_horses.state_dict(), './exported_models/generator_horses%s.pth' % suffix)
    torch.save(mean_net_horses.state_dict(), './exported_models/netMean_horses%s.pth' % suffix)
    torch.save(var_net_horses.state_dict(), './exported_models/netVar_horses%s.pth' % suffix)
    torch.save(generator_zebras.state_dict(), './exported_models/generator_zebras%s.pth' % suffix)
    torch.save(mean_net_zebras.state_dict(), './exported_models/netMean_zebras%s.pth' % suffix)
    torch.save(var_net_zebras.state_dict(), './exported_models/netVar_zebras%s.pth' % suffix)


def sample_images(batch_h, batch_z, num):
    generator_zebras.eval()
    test_gen_z = generator_zebras(batch_h)
    vutils.save_image(test_gen_z.data[:16], './generated_samples/zebras/%d.png' % num, normalize=True)
    generator_zebras.train()

    generator_horses.eval()
    test_gen_h = generator_horses(batch_z)
    vutils.save_image(test_gen_h.data[:16], './generated_samples/horses/%d.png' % num, normalize=True)
    generator_horses.train()


def extract_features_from_batch(batch):
    feats = []
    vgg_out = vgg_pretrained(batch)
    for j in range(batch.size(0)):
        ft_sample = torch.cat([ft[j, :] for ft in vgg_out], dim=0)
        feats.append(ft_sample.view(1, -1))
    return torch.cat(feats, dim=0)

real_mean_horses = torch.load('./data/mean_horses.pt') if os.path.exists('./data/mean_horses.pt') else None
#real_mean_horses = torch.zeros(total_features) ## Remove for real code
real_sqr_horses = None
real_var_horses = torch.load('./data/var_horses.pt') if os.path.exists('./data/var_horses.pt') else None
#real_var_horses = torch.zeros(total_features) ## Remove for real code
num_processed = 0.0
if real_mean_horses is None:
    with torch.no_grad():
        for i, data in tqdm(enumerate(trainloader_horses, 1)):
            img_batch, _ = data
            img_batch = img_batch.to(device)
            extracted_batch = extract_features_from_batch(img_batch)

            if real_mean_horses is None:
                real_mean_horses = torch.sum(extracted_batch, dim=0)
                real_sqr_horses = torch.sum(extracted_batch ** 2, dim=0)
            else:
                real_mean_horses += torch.sum(extracted_batch, dim=0)
                real_sqr_horses += torch.sum(extracted_batch ** 2, dim=0)

            num_processed += img_batch.size(0)

        real_var_horses = (real_sqr_horses - (real_mean_horses ** 2) / num_processed) / (
                num_processed - 1)
        print('normalizing by %.2f' % num_processed)
        real_mean_horses = real_mean_horses / num_processed

        torch.save(real_mean_horses, './data/mean_horses.pt')
        torch.save(real_var_horses, './data/var_horses.pt')

real_mean_zebras = torch.load('./data/mean_zebras.pt') if os.path.exists('./data/mean_zebras.pt') else None
#real_mean_zebras = torch.zeros(total_features)
real_sqr_zebras = None
real_var_zebras = torch.load('./data/var_zebras.pt') if os.path.exists('./data/var_zebras.pt') else None
#real_var_zebras = torch.zeros(total_features)
num_processed = 0.0
if real_mean_zebras is None:
    with torch.no_grad():
        for i, data in tqdm(enumerate(trainloader_zebras, 1)):
            img_batch, _ = data
            img_batch = img_batch.to(device)
            extracted_batch = extract_features_from_batch(img_batch)

            if real_mean_zebras is None:
                real_mean_zebras = torch.sum(extracted_batch, dim=0)
                real_sqr_zebras = torch.sum(extracted_batch ** 2, dim=0)
            else:
                real_mean_zebras += torch.sum(extracted_batch, dim=0)
                real_sqr_zebras += torch.sum(extracted_batch ** 2, dim=0)

            num_processed += img_batch.size(0)

        real_var_zebras = (real_sqr_zebras - (real_mean_zebras ** 2) / num_processed) / (
                num_processed - 1)
        print('normalizing by %.2f' % num_processed)
        real_mean_zebras = real_mean_zebras / num_processed

        torch.save(real_mean_zebras, './data/mean_zebras.pt')
        torch.save(real_var_zebras, './data/var_zebras.pt')


avrg_g_var_net_horses_loss = 0.0
avrg_g_mean_net_horses_loss = 0.0
avrg_mean_net_horses_loss = 0.0
avrg_var_net_horses_loss = 0.0

avrg_g_var_net_zebras_loss = 0.0
avrg_g_mean_net_zebras_loss = 0.0
avrg_mean_net_zebras_loss = 0.0
avrg_var_net_zebras_loss = 0.0

avrg_total_g_loss_zebras = 0.0
avrg_total_g_loss_horses = 0.0
avrg_cycle_loss = 0.0
avrg_combined_loss = 0.0

save_num = 0
print('~~~~~~~~~~~~Starting Training~~~~~~~~~~~~~~~~')
os.sys.stdout.flush()
for epoch in tqdm(range(NUM_ITERATIONS)):
    for i, data in tqdm(enumerate(zip(trainloader_horses, trainloader_zebras), 1)):
        generator_horses.zero_grad()
        mean_net_horses.zero_grad()
        var_net_horses.zero_grad()
        generator_zebras.zero_grad()
        mean_net_zebras.zero_grad()
        var_net_zebras.zero_grad()
        vgg_pretrained.zero_grad()

        horse_batch = data[0][0].to(device)
        zebra_batch = data[1][0].to(device)

        fake_zebras = generator_zebras(horse_batch)
        #fake_imgs = ((fake_imgs*0.5 + 0.5) - imageNetNormMean) / imageNetNormStd
        fake_zebras = (((fake_zebras + 1) * imageNetNormRange) / 2) + imageNetNormMin
        fake_features_zebras = extract_features_from_batch(fake_zebras)

        recycled_horses = generator_horses(fake_zebras)
        cycle_loss_horses = criterionCycleLoss(recycled_horses, horse_batch)

        fake_horses = generator_zebras(zebra_batch)
        # fake_imgs = ((fake_imgs*0.5 + 0.5) - imageNetNormMean) / imageNetNormStd
        fake_horses = (((fake_horses + 1) * imageNetNormRange) / 2) + imageNetNormMin
        fake_features_horses = extract_features_from_batch(fake_horses)

        recycled_zebras = generator_zebras(fake_horses)
        cycle_loss_zebras = criterionCycleLoss(recycled_zebras, zebra_batch)

        # Horses Update
        fake_mean_horses = torch.mean(fake_features_horses, 0)
        real_fake_difference_mean_horses = real_mean_horses.detach() - fake_mean_horses.detach()
        mean_net_horses_loss = criterionLossL2(mean_net_horses.weight, real_fake_difference_mean_horses.detach().view(1, -1))
        mean_net_horses_loss.backward()
        avrg_mean_net_horses_loss += mean_net_horses_loss.item()
        optimizerM_horses.step()

        fake_var_horses = torch.var(fake_features_horses, 0)
        real_fake_difference_var_horses = real_var_horses.detach() - fake_var_horses.detach()
        var_net_horses_loss = criterionLossL2(var_net_horses.weight, real_fake_difference_var_horses.detach().view(1, -1))
        var_net_horses_loss.backward()
        avrg_var_net_horses_loss += var_net_horses_loss.item()
        optimizerV_horses.step()

        mean_diff_real_horses = mean_net_horses(real_mean_horses.view(1, -1)).detach()
        mean_diff_fake_horses = mean_net_horses(fake_mean_horses.view(1, -1))
        var_diff_real_horses = var_net_horses(real_var_horses.view(1, -1)).detach()
        var_diff_fake_horses = var_net_horses(fake_var_horses.view(1, -1))

        g_mean_net_horses_loss = (mean_diff_real_horses - mean_diff_fake_horses)
        avrg_g_mean_net_horses_loss += g_mean_net_horses_loss.item()

        g_var_net_horses_loss = (var_diff_real_horses - var_diff_fake_horses)
        avrg_g_var_net_horses_loss += g_var_net_horses_loss.item()

        generator_loss_horses = g_mean_net_horses_loss + g_var_net_horses_loss
        avrg_total_g_loss_horses += generator_loss_horses.item()

        #Zebras Update
        fake_mean_zebras = torch.mean(fake_features_zebras, 0)
        real_fake_difference_mean_zebras = real_mean_zebras.detach() - fake_mean_zebras.detach()
        mean_net_zebras_loss = criterionLossL2(mean_net_zebras.weight,
                                               real_fake_difference_mean_zebras.detach().view(1, -1))
        mean_net_zebras_loss.backward()
        avrg_mean_net_zebras_loss += mean_net_zebras_loss.item()
        optimizerM_zebras.step()

        fake_var_zebras = torch.var(fake_features_zebras, 0)
        real_fake_difference_var_zebras = real_var_zebras.detach() - fake_var_zebras.detach()
        var_net_zebras_loss = criterionLossL2(var_net_zebras.weight,
                                              real_fake_difference_var_zebras.detach().view(1, -1))
        var_net_zebras_loss.backward()
        avrg_var_net_zebras_loss += var_net_zebras_loss.item()
        optimizerV_zebras.step()

        mean_diff_real_zebras = mean_net_zebras(real_mean_zebras.view(1, -1)).detach()
        mean_diff_fake_zebras = mean_net_zebras(fake_mean_zebras.view(1, -1))
        var_diff_real_zebras = var_net_zebras(real_var_zebras.view(1, -1)).detach()
        var_diff_fake_zebras = var_net_zebras(fake_var_zebras.view(1, -1))

        g_mean_net_zebras_loss = (mean_diff_real_zebras - mean_diff_fake_zebras)
        avrg_g_mean_net_zebras_loss += g_mean_net_zebras_loss.item()

        g_var_net_zebras_loss = (var_diff_real_zebras - var_diff_fake_zebras)
        avrg_g_var_net_zebras_loss += g_var_net_zebras_loss.item()

        generator_loss_zebras = g_mean_net_zebras_loss + g_var_net_zebras_loss
        avrg_total_g_loss_zebras += generator_loss_zebras.item()

        #Optimization
        total_cycle_loss = CYCLE_LAMBDA * (cycle_loss_horses + cycle_loss_zebras)
        avrg_cycle_loss += total_cycle_loss.item()
        combined_loss = generator_loss_horses + generator_loss_zebras + total_cycle_loss
        combined_loss.backward()
        avrg_combined_loss += combined_loss.item()


        optimizerG_horses.step()
        optimizerG_zebras.step()

        if i % 20 == 0:
            with torch.no_grad():
                sample_images(horse_batch, zebra_batch, save_num)
            save_num += 1

    # saving models/images
    #len is from horses dataset, since it is smaller
    print('Zebras: Loss_G_total: %.6f Loss_Gz: %.6f Loss_GzVar: %.6f Loss_vMean: %.6f Loss_vVar: %.6f\n'
          'Horses: Loss_G_total: %.6f Loss_Gz: %.6f Loss_GzVar: %.6f Loss_vMean: %.6f Loss_vVar: %.6f\n'
          'Combined losses: Cycle_loss: %.6f, Total_loss: %.6f' %
          (avrg_total_g_loss_zebras / len(trainset_horses),
           avrg_g_mean_net_zebras_loss / len(trainset_horses), avrg_g_var_net_zebras_loss / len(trainset_horses),
           avrg_mean_net_zebras_loss / len(trainset_horses), avrg_var_net_zebras_loss / len(trainset_horses),
           avrg_total_g_loss_horses / len(trainset_horses),
           avrg_g_mean_net_horses_loss / len(trainset_horses), avrg_g_var_net_horses_loss / len(trainset_horses),
           avrg_mean_net_horses_loss / len(trainset_horses), avrg_var_net_horses_loss / len(trainset_horses),
           avrg_cycle_loss/len(trainset_horses), avrg_combined_loss/len(trainset_horses)
           ))
    os.sys.stdout.flush()

    avrg_g_var_net_horses_loss = 0.0
    avrg_g_mean_net_horses_loss = 0.0
    avrg_mean_net_horses_loss = 0.0
    avrg_var_net_horses_loss = 0.0

    avrg_g_var_net_zebras_loss = 0.0
    avrg_g_mean_net_zebras_loss = 0.0
    avrg_mean_net_zebras_loss = 0.0
    avrg_var_net_zebras_loss = 0.0

    avrg_total_g_loss_zebras = 0.0
    avrg_total_g_loss_horses = 0.0
    avrg_cycle_loss = 0.0
    avrg_combined_loss = 0.0
    save_models("")

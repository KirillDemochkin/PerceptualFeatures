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
from models.resnet import Resnet18Full
from models.cycle_gan_models import ResnetGenerator, MultiscaleDiscriminator
from models.generator_models import weights_init

BATCH_SIZE = 64
B1 = 0.5
LR_G = 5e-5
LR_MV_AVG = 5e-5
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
# resnet_pretrained = Resnet18Full().to(device).eval()

generator_horses = ResnetGenerator(input_nc=3,
                                   output_nc=3,
                                   ngf=64,
                                   norm_layer=functools.partial(nn.InstanceNorm2d, affine=False,
                                                                track_running_stats=False),
                                   use_dropout=False,
                                   n_blocks=6,
                                   padding_type='reflect').to(device)
generator_zebras = ResnetGenerator(input_nc=3,
                                   output_nc=3,
                                   ngf=64,
                                   norm_layer=functools.partial(nn.InstanceNorm2d, affine=False,
                                                                track_running_stats=False),
                                   use_dropout=False,
                                   n_blocks=6,
                                   padding_type='reflect').to(device)
discriminator_horses = MultiscaleDiscriminator().to(device)
discriminator_zebras = MultiscaleDiscriminator().to(device)

generator_zebras.apply(weights_init)
generator_horses.apply(weights_init)

discriminator_horses.apply(weights_init)
discriminator_zebras.apply(weights_init)

total_features = 0
disc_features = 0
with torch.no_grad():
    empty_res_vgg = vgg_pretrained(torch.empty(4, 3, IMG_SIZE, IMG_SIZE).normal_(mean=0, std=1).to(device))
    # empty_res_resnet = resnet_pretrained(torch.empty(4, 3, IMG_SIZE, IMG_SIZE).normal_(mean=0, std=1).to(device))
    _, empty_res_disc = discriminator_horses(torch.empty(4, 3, IMG_SIZE, IMG_SIZE).normal_(mean=0, std=1).to(device))
    print([er.shape[1] for er in empty_res_vgg])
    # print([er.shape[1] for er in empty_res_resnet])
    print([er.shape[1] for er in empty_res_disc])
for r in empty_res_vgg:
    total_features += r.shape[1]
# for r in empty_res_resnet:
# total_features += r.shape[1]
for r in empty_res_disc:
    total_features += r.shape[1]
    disc_features += r.shape[1]
print('Performing feature matching for %d features' % total_features)
# mean/var nets, losses, and optimizers
mean_net_horses = nn.Linear(total_features, 1, bias=False).to(device)
var_net_horses = nn.Linear(total_features, 1, bias=False).to(device)

mean_net_zebras = nn.Linear(total_features, 1, bias=False).to(device)
var_net_zebras = nn.Linear(total_features, 1, bias=False).to(device)

mean_net_horses_disc = nn.Linear(disc_features, 1, bias=False).to(device)
var_net_horses_disc = nn.Linear(disc_features, 1, bias=False).to(device)

mean_net_zebras_disc = nn.Linear(disc_features, 1, bias=False).to(device)
var_net_zebras_disc = nn.Linear(disc_features, 1, bias=False).to(device)

criterionLossL2 = nn.MSELoss().to(device)
criterionCycleLoss = nn.L1Loss().to(device)

parametersG_horses = set()
parametersG_horses |= set(generator_horses.parameters())
optimizerG_horses = optim.Adam(parametersG_horses, LR_G, betas=(B1, 0.999))
optimizerM_horses = optim.Adam(mean_net_horses.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerV_horses = optim.Adam(var_net_horses.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerMD_horses = optim.Adam(mean_net_horses_disc.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerVD_horses = optim.Adam(var_net_horses_disc.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerD_horses = optim.Adam(discriminator_horses.parameters(), LR_G, betas=(B1, 0.999))

parametersG_zebras = set()
parametersG_zebras |= set(generator_zebras.parameters())
optimizerG_zebras = optim.Adam(parametersG_zebras, LR_G, betas=(B1, 0.999))
optimizerM_zebras = optim.Adam(mean_net_zebras.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerV_zebras = optim.Adam(var_net_zebras.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerMD_zebras = optim.Adam(mean_net_zebras_disc.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerVD_zebras = optim.Adam(var_net_zebras_disc.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerD_zebras = optim.Adam(discriminator_zebras.parameters(), LR_G, betas=(B1, 0.999))


def save_models(suffix=""):
    # saving current best model
    torch.save(generator_horses.state_dict(), './exported_models/generator_horses%s.pth' % suffix)
    torch.save(mean_net_horses.state_dict(), './exported_models/netMean_horses%s.pth' % suffix)
    torch.save(var_net_horses.state_dict(), './exported_models/netVar_horses%s.pth' % suffix)
    torch.save(mean_net_horses_disc.state_dict(), './exported_models/netMean_horses_disc%s.pth' % suffix)
    torch.save(var_net_horses_disc.state_dict(), './exported_models/netVar_horses_disc%s.pth' % suffix)
    torch.save(generator_zebras.state_dict(), './exported_models/generator_zebras%s.pth' % suffix)
    torch.save(mean_net_zebras.state_dict(), './exported_models/netMean_zebras%s.pth' % suffix)
    torch.save(var_net_zebras.state_dict(), './exported_models/netVar_zebras%s.pth' % suffix)
    torch.save(mean_net_zebras_disc.state_dict(), './exported_models/netMean_zebra_disc%s.pth' % suffix)
    torch.save(var_net_zebras_disc.state_dict(), './exported_models/netVar_zebras_disc%s.pth' % suffix)


def sample_images(batch_h, batch_z, num):
    generator_zebras.eval()
    test_gen_z = generator_zebras(batch_h)
    vutils.save_image(test_gen_z.data[:16], './generated_samples/zebras/%d.png' % num, normalize=True)
    generator_zebras.train()

    generator_horses.eval()
    test_gen_h = generator_horses(batch_z)
    vutils.save_image(test_gen_h.data[:16], './generated_samples/horses/%d.png' % num, normalize=True)
    generator_horses.train()


def extract_features_from_batch(batch, disc_feats=None):
    feats = []
    discr_feats = []
    vgg_out = vgg_pretrained(batch)
    # resnet_out = resnet_pretrained(batch)
    for j in range(batch.size(0)):
        ft_sample_vgg = torch.cat([ft[j, :] for ft in vgg_out], dim=0).view(1, -1)
        # ft_sample_resnet = torch.cat([ft[j, :] for ft in resnet_out], dim=0).view(1, -1)
        if disc_feats is not None:
            ft_sample_disc = torch.cat([ft[j, :] for ft in disc_feats], dim=0).view(1, -1)
        # ft_sample = torch.cat((ft_sample_vgg, ft_sample_resnet, ft_sample_disc), dim=1)
        if disc_feats is not None:
            ft_sample = torch.cat((ft_sample_vgg, ft_sample_disc), dim=1)
        else:
            ft_sample = ft_sample_vgg
        feats.append(ft_sample)
        if disc_feats is not None:
            discr_feats.append(ft_sample_disc)
    return torch.cat(feats, dim=0)


real_mean_horses = torch.load('./data/mean_horses.pt') if os.path.exists('./data/mean_horses.pt') else None
# real_mean_horses = torch.zeros(total_features) ## Remove for real code
real_sqr_horses = None
real_var_horses = torch.load('./data/var_horses.pt') if os.path.exists('./data/var_horses.pt') else None
# real_var_horses = torch.zeros(total_features) ## Remove for real code
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
# real_mean_zebras = torch.zeros(total_features)
real_sqr_zebras = None
real_var_zebras = torch.load('./data/var_zebras.pt') if os.path.exists('./data/var_zebras.pt') else None
# real_var_zebras = torch.zeros(total_features)
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


def hinge_loss_discriminator(fake_preds, real_preds):
    rpl = torch.min(real_preds - 1, torch.zeros_like(real_preds))
    fpl = torch.min(-fake_preds - 1, torch.zeros_like(fake_preds))
    return -torch.mean(
        torch.add(rpl, fpl))


def hinge_loss_generator(fake_preds):
    return -torch.mean(fake_preds)


print('~~~~~~~~~~~~Starting Training~~~~~~~~~~~~~~~~')
os.sys.stdout.flush()
for epoch in tqdm(range(NUM_ITERATIONS)):
    for i, data in tqdm(enumerate(zip(trainloader_horses, trainloader_zebras), 1)):
        discriminator_horses.zero_grad()
        discriminator_zebras.zero_grad()

        horse_batch = data[0][0].to(device)
        zebra_batch = data[1][0].to(device)
        jitter_real_h = torch.empty_like(horse_batch, device=device).uniform_(-0.05 * (0.99 ** epoch),
                                                                            0.05 * (0.99 ** epoch))
        jitter_real_z = torch.empty_like(zebra_batch, device=device).uniform_(-0.05 * (0.99 ** epoch),
                                                                            0.05 * (0.99 ** epoch))
        jitter_fake_h = torch.empty_like(horse_batch, device=device).uniform_(-0.05 * (0.99 ** epoch),
                                                                            0.05 * (0.99 ** epoch))
        jitter_fake_z = torch.empty_like(zebra_batch, device=device).uniform_(-0.05 * (0.99 ** epoch),
                                                                            0.05 * (0.99 ** epoch))

        real_horses_preds, real_horses_disc_feats = discriminator_horses(torch.clamp(horse_batch + jitter_real_h, -1, 1))
        real_zebras_preds, real_zebras_disc_feats = discriminator_zebras(torch.clamp(zebra_batch + jitter_real_z, -1, 1))

        fake_zebras = generator_zebras(horse_batch)
        fake_zebras_preds, fake_zebras_disc_feats = discriminator_zebras(
            torch.clamp(fake_zebras.detach() + jitter_fake_z, -1, 1))
        fake_zebras_normalized = (((fake_zebras + 1) * imageNetNormRange) / 2) + imageNetNormMin
        recycled_horses = generator_horses(fake_zebras_normalized)
        # fake_imgs = ((fake_imgs*0.5 + 0.5) - imageNetNormMean) / imageNetNormStd

        fake_horses = generator_horses(zebra_batch)
        # fake_imgs = ((fake_imgs*0.5 + 0.5) - imageNetNormMean) / imageNetNormStd
        fake_horses_preds, fake_horses_disc_feats = discriminator_horses(
            torch.clamp(fake_horses.detach() + jitter_fake_h, -1, 1))
        fake_horses_normalized = (((fake_horses + 1) * imageNetNormRange) / 2) + imageNetNormMin
        recycled_zebras = generator_zebras(fake_horses_normalized)

        errDz = 0.0
        for fp, rp in zip(fake_zebras_preds, real_zebras_preds):
            errDz += hinge_loss_discriminator(fp, rp)
        errDz.backward()
        optimizerD_zebras.step()

        errDh = 0.0
        for fp, rp in zip(fake_horses_preds, real_horses_preds):
            errDh += hinge_loss_discriminator(fp, rp)
        errDh.backward()
        optimizerD_horses.step()

        generator_horses.zero_grad()
        mean_net_horses.zero_grad()
        var_net_horses.zero_grad()
        mean_net_horses_disc.zero_grad()
        var_net_horses_disc.zero_grad()
        generator_zebras.zero_grad()
        mean_net_zebras.zero_grad()
        var_net_zebras.zero_grad()
        mean_net_zebras_disc.zero_grad()
        var_net_zebras_disc.zero_grad()
        vgg_pretrained.zero_grad()
        # resnet_pretrained.zero_grad()

        fake_horses_preds, fake_horses_disc_feats = discriminator_horses(fake_horses)
        fake_zebras_preds, fake_zebras_disc_feats = discriminator_zebras(fake_zebras)

        fake_features_zebras = extract_features_from_batch(fake_zebras_normalized, fake_zebras_disc_feats)
        fake_features_horses = extract_features_from_batch(fake_horses_normalized, fake_horses_disc_feats)

        cycle_loss_zebras = criterionCycleLoss(recycled_zebras, zebra_batch)
        cycle_loss_horses = criterionCycleLoss(recycled_horses, horse_batch)

        # Horses Update
        fake_mean_horses = torch.mean(fake_features_horses, 0)
        #print(fake_mean_horses.shape)
        #print(torch.mean(torch.cat(real_horses_disc_feats, dim=1), dim=0, keepdim=True).shape)
        real_mean_horses_discr = torch.cat((real_mean_horses,
                                            torch.mean(torch.cat(real_horses_disc_feats, dim=1), dim=0)),
                                            dim=0)
        real_fake_difference_mean_horses = real_mean_horses_discr.detach() - fake_mean_horses.detach()
        mean_net_horses_loss = criterionLossL2(mean_net_horses.weight,
                                               real_fake_difference_mean_horses.detach().view(1, -1))
        mean_net_horses_loss.backward()
        avrg_mean_net_horses_loss += mean_net_horses_loss.item()
        optimizerM_horses.step()

        fake_var_horses = torch.var(fake_features_horses, 0, keepdim=True)
        real_var_horses_discr = torch.cat((real_var_horses,
                                           torch.var(torch.cat(real_horses_disc_feats, dim=1), dim=0)),
                                           dim=0)
        real_fake_difference_var_horses = real_var_horses_discr.detach() - fake_var_horses.detach()
        var_net_horses_loss = criterionLossL2(var_net_horses.weight,
                                              real_fake_difference_var_horses.detach().view(1, -1))
        var_net_horses_loss.backward()
        avrg_var_net_horses_loss += var_net_horses_loss.item()
        optimizerV_horses.step()

        mean_diff_real_horses = mean_net_horses(real_mean_horses_discr.view(1, -1)).detach()
        mean_diff_fake_horses = mean_net_horses(fake_mean_horses.view(1, -1))
        var_diff_real_horses = var_net_horses(real_var_horses_discr.view(1, -1)).detach()
        var_diff_fake_horses = var_net_horses(fake_var_horses.view(1, -1))

        g_mean_net_horses_loss = (mean_diff_real_horses - mean_diff_fake_horses)
        avrg_g_mean_net_horses_loss += g_mean_net_horses_loss.item()

        g_var_net_horses_loss = (var_diff_real_horses - var_diff_fake_horses)
        avrg_g_var_net_horses_loss += g_var_net_horses_loss.item()

        generator_loss_horses = g_mean_net_horses_loss + g_var_net_horses_loss
        avrg_total_g_loss_horses += generator_loss_horses.item()

        # Zebras Update
        fake_mean_zebras = torch.mean(fake_features_zebras, 0)
        real_mean_zebras_discr = torch.cat((real_mean_zebras,
                                            torch.mean(torch.cat(real_zebras_disc_feats, dim=1), 0)),
                                            dim=0)
        real_fake_difference_mean_zebras = real_mean_zebras_discr.detach() - fake_mean_zebras.detach()
        mean_net_zebras_loss = criterionLossL2(mean_net_zebras.weight,
                                               real_fake_difference_mean_zebras.detach().view(1, -1))
        mean_net_zebras_loss.backward()
        avrg_mean_net_zebras_loss += mean_net_zebras_loss.item()
        optimizerM_zebras.step()

        fake_var_zebras = torch.var(fake_features_zebras, 0)
        real_var_zebras_discr = torch.cat((real_var_zebras,
                                           torch.var(torch.cat(real_zebras_disc_feats, dim=1), 0)),
                                           dim=0)
        real_fake_difference_var_zebras = real_var_zebras_discr.detach() - fake_var_zebras.detach()
        var_net_zebras_loss = criterionLossL2(var_net_zebras.weight,
                                              real_fake_difference_var_zebras.detach().view(1, -1))
        var_net_zebras_loss.backward()
        avrg_var_net_zebras_loss += var_net_zebras_loss.item()
        optimizerV_zebras.step()

        mean_diff_real_zebras = mean_net_zebras(real_mean_zebras_discr.view(1, -1)).detach()
        mean_diff_fake_zebras = mean_net_zebras(fake_mean_zebras.view(1, -1))
        var_diff_real_zebras = var_net_zebras(real_var_zebras_discr.view(1, -1)).detach()
        var_diff_fake_zebras = var_net_zebras(fake_var_zebras.view(1, -1))

        g_mean_net_zebras_loss = (mean_diff_real_zebras - mean_diff_fake_zebras)
        avrg_g_mean_net_zebras_loss += g_mean_net_zebras_loss.item()

        g_var_net_zebras_loss = (var_diff_real_zebras - var_diff_fake_zebras)
        avrg_g_var_net_zebras_loss += g_var_net_zebras_loss.item()

        generator_loss_zebras = g_mean_net_zebras_loss + g_var_net_zebras_loss
        avrg_total_g_loss_zebras += generator_loss_zebras.item()

        # Optimization
        errGh = 0.0
        for fp in fake_horses_preds:
            errGh += hinge_loss_generator(fp)
        errGh.backward(retain_graph=True)

        errGz = 0.0
        for fp in fake_zebras_preds:
            errGz += hinge_loss_generator(fp)
        errGz.backward(retain_graph=True)

        total_cycle_loss = CYCLE_LAMBDA * (cycle_loss_horses + cycle_loss_zebras)
        avrg_cycle_loss += total_cycle_loss.item()
        combined_loss = generator_loss_horses + generator_loss_zebras + total_cycle_loss
        combined_loss.backward()
        avrg_combined_loss += combined_loss.item() + errGz.item() + errGh.item()

        optimizerG_horses.step()
        optimizerG_zebras.step()

        if i % 50 == 49:
            with torch.no_grad():
                sample_images(horse_batch, zebra_batch, save_num)
            save_num += 1

    # saving models/images
    # len is from horses dataset, since it is smaller
    print('Zebras: Loss_G_total: %.6f Loss_Gz: %.6f Loss_GzVar: %.6f Loss_vMean: %.6f Loss_vVar: %.6f\n'
          'Horses: Loss_G_total: %.6f Loss_Gz: %.6f Loss_GzVar: %.6f Loss_vMean: %.6f Loss_vVar: %.6f\n'
          'Combined losses: Cycle_loss: %.6f, Total_loss: %.6f' %
          (avrg_total_g_loss_zebras / len(trainset_horses),
           avrg_g_mean_net_zebras_loss / len(trainset_horses), avrg_g_var_net_zebras_loss / len(trainset_horses),
           avrg_mean_net_zebras_loss / len(trainset_horses), avrg_var_net_zebras_loss / len(trainset_horses),
           avrg_total_g_loss_horses / len(trainset_horses),
           avrg_g_mean_net_horses_loss / len(trainset_horses), avrg_g_var_net_horses_loss / len(trainset_horses),
           avrg_mean_net_horses_loss / len(trainset_horses), avrg_var_net_horses_loss / len(trainset_horses),
           avrg_cycle_loss / len(trainset_horses), avrg_combined_loss / len(trainset_horses)
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

import torch
import torch.nn as nn
import torch.optim as optim
from models.vgg import Vgg16
from models.generator_models import DCGAN
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
netEnc = []

VGG_PATH = "./exported_models/vgg19.pt"
LATENT_DIM = 100
B1 = 0.5
LR_G = 0.002
LR_MV_AVG = 0.002

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

imageNetNormMean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
imageNetNormStd = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
imageNetNormMin = -imageNetNormMean / imageNetNormStd
imageNetNormMax = (1.0 - imageNetNormMean) / imageNetNormStd
imageNetNormRange = imageNetNormMax - imageNetNormMin

imageNetNormMin = torch.tensor(imageNetNormMin, dtype=torch.float32).to(device)
imageNetNormMin.resize_(1, 3, 1, 1)
imageNetNormRange = torch.tensor(imageNetNormRange, dtype=torch.float32).to(device)
imageNetNormRange.resize_(1, 3, 1, 1)

transform = [transforms.ToTensor()]

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

test_noise = torch.empty(16, LATENT_DIM, 1, 1).normal_(mean=0, std=1).to(device)

print("Loading VGG")
vgg_pretrained = Vgg16()
vgg_pretrained.to(device)
vgg_pretrained.eval()

generator = DCGAN(LATENT_DIM)
generator.to(device)

# normalize tensors
# count total features
total_features = 0
with torch.no_grad():
    empty_res = vgg_pretrained(torch.empty(4, 3, 32, 32).normal_(mean=0, std=1).to(device))
for r in empty_res:
    total_features += r.shape[1]
print('Performing feature mathcing for %d features' % total_features)
# mean/var nets, losses, and optimizers
mean_net = nn.Linear(total_features, 1, bias=False)
mean_net.to(device)

var_net = nn.Linear(total_features, 1, bias=False)
var_net.to(device)

criterionLossL1 = nn.L1Loss()
criterionLossL2 = nn.MSELoss()

optimizerG = optim.Adam(generator.parameters(), LR_G, betas=(B1, 0.999))
optimizerM = optim.Adam(mean_net.parameters(), LR_MV_AVG, betas=(B1, 0.999))
optimizerV = optim.Adam(var_net.parameters(), LR_MV_AVG, betas=(B1, 0.999))
# feature extraction


def extract_features_from_batch(batch):
    feats = []
    vgg_out = vgg_pretrained(batch)
    for i in range(batch.size(0)):
        ft_sample = torch.cat([ft[i] for ft in vgg_out], dim=0)
        feats.append(ft_sample.view(1, -1))
    return torch.cat(feats, dim=0)


with torch.no_grad():
    extracted = extract_features_from_batch(torch.empty(4, 3, 32, 32).normal_(mean=0, std=1).to(device))
    print(extracted.shape)
# training loop
# saving models/images





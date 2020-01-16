import torch
import torch.nn as nn
import torch.optim as optim
from models.vgg import Vgg16Full
from models.generator_models import DCGAN
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
netEnc = []

VGG_PATH = "./exported_models/vgg19.pt"
LATENT_DIM = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

imageNetNormMean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
imageNetNormStd = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
imageNetNormMin = -imageNetNormMean / imageNetNormStd
imageNetNormMax = (1.0 - imageNetNormMean) / imageNetNormStd
imageNetNormRange = imageNetNormMax - imageNetNormMin

imageNetNormMin = torch.tensor(imageNetNormMin, dtype=torch.float32)
imageNetNormRange = torch.tensor(imageNetNormRange, dtype=torch.float32)

transform = [transforms.ToTensor()]

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)


test_noise = torch.empty(16, 1, 1, LATENT_DIM).normal_(mean=0, std=1).to(device)

print("Loading VGG")
vgg_pretrained = Vgg16Full()
vgg_pretrained.to(device)

generator = DCGAN(LATENT_DIM)
generator.to(device)

testres = generator(test_noise)

# normalize tensors
# count total features
# mean/var nets, losses, and optimizers
# training loop
# saving models/images

print(testres.shape)




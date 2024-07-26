from unet_model import UNet_Var
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from medmnist.dataset import OrganAMNIST
import os
import json
import argparse

# define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# define dataset
train_dataset = OrganAMNIST(root='./data', split='train', transform=transform, download=True, size=224)
test_dataset = OrganAMNIST(root='./data', split='test', transform=transform, download=True, size=224)

# define dataloader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# check dataset size
print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))

for i, (img, label) in enumerate(train_loader):
    print(f"Batch {i} - Image shape: {img.shape}, Label shape: {label.shape}")
    break
# define model
# model = UNet_Var(1, 1, 11)

# output = model(test_tensor)
# output, cls_out = output
# print(output.shape, cls_out.shape)
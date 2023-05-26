from inspect import isfunction
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.datasets import Flowers102
import numpy as np



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# exact the number alphas, betas, ... on the right timestep needed.
def extract(a, t, x_shape):
    # batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    # print(out.shape)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1))).to(t.device)

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# data

def get_reverse_transform():
     return Compose([
          Lambda(lambda t: (t + 1) / 2), # transform pixel range to (0, 1)
          Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
          Lambda(lambda t: t * 255.),
          Lambda(lambda t: t.numpy().astype(np.uint8)),
          ToPILImage(),
     ])
def get_transform():
     return Compose([
          transforms.RandomHorizontalFlip(),
          transforms.Resize(size=(64, 64)),
          transforms.ToTensor(),
          transforms.Lambda(lambda t: (t * 2) - 1) # pixel range (-1, 1)
          ])


def get_data():
     train_dataset = Flowers102(root="/home/anhtt1/workspace/phuclh15/Diffusion_model/dataset/flower_dataset",
                      split='train', 
                      transform=get_transform(),
                      download=True)
     dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
     return dataloader
from inspect import isfunction
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.datasets import Flowers102
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import wandb

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# exact the number alphas, betas, ... on the right timestep needed.

def cycle(dl):
    while True:
        for data in dl:
            yield data

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


def get_reverse_transform():
    return Compose([
        Lambda(lambda t: (t + 1) / 2),  # transform pixel range to (0, 1)
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])


def get_transform():
    return Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # pixel range (-1, 1)
    ])


def get_train_dataloader(data_path, batch_size):
    train_dataset = Flowers102(root=data_path,
                               split='train',
                               transform=get_transform(), download=True)
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

def get_train_dataloader_debug(data_path, batch_size):
    train_dataset = Flowers102(root=data_path,
                               split='train',
                               transform=get_transform(), download=True)
    sampler = SubsetRandomSampler(range(2 * batch_size))
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    return dataloader

def get_val_dataloader(data_path, batch_size):
    val_dataset = Flowers102(root=data_path,
                             split='val',
                             transform=get_transform(), download=True)
    dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader


def get_test_dataloader(data_path, batch_size):
    test_dataset = Flowers102(root=data_path,
                              split='test',
                              transform=get_transform(), download=True)
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

# function to save checkpoint. It can be used to save to WandB or local.


def save_checkpoint(model, optimizer, scheduler, args, path, steps, save_to_wandb=False):
    ckpt_path = path + f"/checkpoint_step_{steps}.pt"
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        'args': args
    }
    torch.save(checkpoint, ckpt_path)
    if save_to_wandb:
        wandb.save(ckpt_path, base_path="./checkpoints/")
    print("saved checkpoint to {}".format(ckpt_path))

# function to load checkpoint. It can be used to load from WandB or local.
def load_checkpoint_local(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler.load_state_dict(checkpoint['scheduler'])
    args = checkpoint['args']
    return model, optimizer, scheduler, args
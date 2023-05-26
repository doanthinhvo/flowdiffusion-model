import os
import torch
import torch.nn as nn
# from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import Unet
import logging
from ddpm import Diffusion
from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

def train(args):
    # setup_logging(args.run_name)
    device = args.device
    dataloader = get_data()
    model = model = Unet(
        dim=args.image_size,
        channels=args.channels,
        dim_mults=(1, 2, 4, 8) 
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    # l = len(dataloader)
    num_step_per_epoch = len(dataloader)
    save_and_sample_every = 1000
    
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok = True)
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}: ")
        pbar = tqdm(dataloader)
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            batch,_ = batch
            batch = batch.to(device)
            batch_size = batch.shape[0]
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, args.timesteps, (batch_size,), device=device).long()

            loss = diffusion.p_losses(model, batch, t, loss_type="l2")

            print("Epoch : [%d], iter [%d], Loss: [%f]" %(epoch, step, loss.item()))

            loss.backward()
            optimizer.step()
            step = step + epoch * num_step_per_epoch
            # save generated images
            if step == 0 or step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                num_images_sample = 32
                all_images_list = diffusion.sample(model, args.image_size, batch_size=num_images_sample, channels=args.channels)
                all_images = all_images_list[-1]
                all_images = (all_images + 1) * 0.5
                save_image(torch.from_numpy(all_images), str(results_folder / f'sample-{milestone}.png'), nrow = 4)

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 8
    args.channels = 3
    args.image_size = 64
    args.dataset_path = r"/kaggle/input/resized-oxfords-flower-dataset/resized_jpg"
    args.device = "cuda"
    args.lr = 3e-4
    args.timesteps=1000
    train(args)

if __name__ == '__main__':
    launch()
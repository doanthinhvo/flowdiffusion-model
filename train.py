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
from datetime import datetime

def train(args):
    # TODO: Setup Logging.
    # setup_logging(args.run_name)
    dataloader = get_data(args.data_path, args.batch_size)
    model = Unet(
        dim=args.image_size,
        channels=args.channels,
        dim_mults=(1, 2, 4, 8) 
    ).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=args.device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    # l = len(dataloader)
    num_step_per_epoch = len(dataloader)
    save_and_sample_every = 200
    
    results_folder = Path(args.results_path)
    results_folder.mkdir(exist_ok = True)
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}: ")
        # pbar = tqdm(dataloader)
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch,_ = batch
            batch = batch.to(args.device)
            batch_size = batch.shape[0]
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, args.timesteps, (batch_size,), device=args.device).long()

            loss = diffusion.p_losses(model, batch, t, loss_type="l2")

            loss.backward()
            optimizer.step()
            step = step + epoch * num_step_per_epoch
            
            print("Epoch [%d], iter [%d], Loss: [%f]" %(epoch, step, loss.item()))
            
            # save generated images
            overal_steps = step + epoch * len(dataloader)
            
            if step == 0 or overal_steps % save_and_sample_every == 0:
                # milestone = step // save_and_sample_every
                num_images_sample = 32
                all_images_list = diffusion.sample(model, args.image_size, batch_size=num_images_sample, channels=args.channels)
                all_images = all_images_list[-1]
                all_images = (all_images + 1) * 0.5
                save_image(torch.from_numpy(all_images), str(results_folder / f'sample-{overal_steps}.png'), nrow = 4)
                print("Images saved \n")

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 128
    args.channels = 3
    args.image_size = 64
    args.data_path = "/home/anhtt1/workspace/phuclh15/Diffusion_model/dataset/flower_dataset"
    args.device = "cuda"
    args.lr = 8e-5
    args.timesteps=1000
    args.results_path = f"/home/anhtt1/workspace/phuclh15/THINH/denoising_diffusion_new/results/{datetime.now().day}-{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}"
    train(args)

if __name__ == '__main__':
    launch()
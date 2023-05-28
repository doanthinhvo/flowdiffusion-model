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
# import wandb
from PIL import Image

def train(args):

    wandb.init(
    project="flower_diffsion",
    config=args
    )
    
    dl = get_train_dataloader(args.data_path, args.batch_size)
    dataloader = cycle(dl)

    model = Unet(
        dim=args.image_size,
        channels=args.channels,
        dim_mults=(1, 2, 4, 8) 
    ).to(args.device)

    scheduler=None
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = Diffusion(timesteps=args.timesteps, img_size=args.image_size, device=args.device)
    results_folder = Path(args.results_path)
    results_folder.mkdir(exist_ok = True)
    
    # load checkpoints if they exist
    if args.checkpoint_path is not None:
        if os.path.isfile(args.checkpoint_path):
            logging.info(f"Checkpoint found at {args.checkpoint_path}")
            print(f"Checkpoint found at {args.checkpoint_path}")
            model, optimizer, scheduler, args = load_checkpoint_local(model,optimizer, scheduler, args.checkpoint_path)
        else:
            logging.info(f"No checkpoint found at {args.checkpoint_path}")

    # if args.wandb_save_model:
    #     logging.info(f"Restoring from wandb")
    #     print(f"Restoring from wandb")
    #     model, optimizer, scheduler, args = load_checkpoint_wandb(model, optimizer, scheduler, args)
        
    overal_steps = 0
    while overal_steps <= args.train_num_steps:

        data = next(dataloader)
        optimizer.zero_grad()
        batch,_ = data
        batch = batch.to(args.device)
        batch_size = batch.shape[0]
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, args.timesteps, (batch_size,), device=args.device).long()

        loss = diffusion.p_losses(model, batch, t, loss_type="l2")

        wandb.log({"loss": loss}, step=overal_steps)

        loss.backward() # compute the gradients
        optimizer.step() # update the parameters.parameters still be on the model. call it by model.parameters()

        print("Step [%d], Loss: [%f]" %(overal_steps, loss.item()))
        
        if overal_steps != 0 and overal_steps % args.save_and_sample_every == 0:
            num_images_sample = 32
            all_images_list = diffusion.sample(model, args.image_size, batch_size=num_images_sample, channels=args.channels)
            all_images = all_images_list[-1]
            all_images = (all_images + 1) * 0.5
            save_image(torch.from_numpy(all_images), str(results_folder / f'sample-{overal_steps}.png'), nrow = 4)
            
            wandb_image = wandb.Image(Image.open(str(results_folder / f'sample-{overal_steps}.png')), caption=f"Step {overal_steps}")
            wandb.log({" ---> sample image ": wandb_image}, step=overal_steps)
            print("========== Sample images and Checkpoint are saved ========== \n")
            save_checkpoint(model, optimizer, scheduler, args, args.checkpoint_path, overal_steps, args.save_to_wandb)
        overal_steps += 1
    wandb.finish()

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.save_to_wandb = True
    args.debug = True
    args.run_name = "DDPM_Uncondtional"
    args.timesteps = 3
    args.train_num_steps = 10
    args.batch_size = 7
    args.channels = 3
    args.image_size = 64
    # args.data_path = "/media/doanthinhvo/OS/Users/doant/Downloads/flowdiffusion-model/data/"
    args.data_path = "/kaggle/working/flowdiffusion-model/data/"
    args.device = "cuda"
    args.lr = 8e-5
    # args.results_path = f"/media/doanthinhvo/OS/Users/doant/Downloads/flowdiffusion-model/results/{datetime.now().day}-{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}/"
    args.results_path = f"/kaggle/working/flowdiffusion-model/results/{datetime.now().day}-{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}/"    
    args.save_and_sample_every = 3
    # args.checkpoint_path = "/media/doanthinhvo/OS/Users/doant/Downloads/flowdiffusion-model/checkpoints/"
    args.checkpoint_path = f"/kaggle/working/flowdiffusion-model/checkpoints/"
    args.wandb_save_model=True
    train(args)

if __name__ == '__main__':
    wandb.login(key="4d3dc8e91e95378b0774701cb2940c1b574bc368")
    launch()
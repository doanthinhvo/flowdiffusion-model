import os
import torch
import torch.nn as nn
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
import wandb
from PIL import Image
from calculate_fid_score import *

def train(args):
    # wandb
    if args.save_to_wandb:
        wandb.login(key="4d3dc8e91e95378b0774701cb2940c1b574bc368")
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

    fid_score = FIDScore(args.device)

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

        print("Step [%d] - Loss: [%f]" %(overal_steps, loss.item()))

        if args.save_to_wandb:
            wandb.log({"loss": loss}, step=overal_steps)

        loss.backward() # compute the gradients
        optimizer.step() # update the parameters.parameters still be on the model. call it by model.parameters()
        
        if overal_steps != 0 and overal_steps % args.save_and_sample_every == 0:
            num_images_sample = 32
            all_images_list = diffusion.sample(model, args.image_size, batch_size=num_images_sample, channels=args.channels)
            all_images = all_images_list[-1]

            print(f"Step {overal_steps} - FID score: {fid_score.fid_score(batch, torch.from_numpy(all_images).to(args.device))}")
            all_images = (all_images + 1) * 0.5
            save_image(torch.from_numpy(all_images), str(results_folder / f'sample-{overal_steps}.png'), nrow = 4)
            
            if args.save_to_wandb:
                wandb_image = wandb.Image(Image.open(str(results_folder / f'sample-{overal_steps}.png')), caption=f"Step {overal_steps}")
                wandb.log({"---> sample image ": wandb_image}, step=overal_steps)

            print("========== sample images and checkpoint are saved ========== \n")
            save_checkpoint(model, optimizer, scheduler, args, args.checkpoint_path, overal_steps, args.save_to_wandb)
        overal_steps += 1
    wandb.finish()

def launch(debug=False):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.run_name = "DDPM_Uncondtional"

    # Experiment
    args.save_to_wandb = False
    args.kaggle = False

    # DEBUG
    if debug:
        args.timesteps = 3
        args.train_num_steps = 5
        args.batch_size = 8
        args.save_and_sample_every = 1
    else:
        args.timesteps = 1000
        args.train_num_steps = 20000
        args.batch_size = 8
        args.save_and_sample_every = 1000

    # Dataset.
    if args.kaggle:
        args.data_path = "/kaggle/working/flowdiffusion-model/data/"
        args.results_path = f"/kaggle/working/flowdiffusion-model/results/{datetime.now().day}-{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}/"    
        args.checkpoint_path = f"/kaggle/working/flowdiffusion-model/checkpoints/"
    else:
        args.data_path = "/media/doanthinhvo/OS/Users/doant/Downloads/flowdiffusion-model/data/"
        args.results_path = f"/media/doanthinhvo/OS/Users/doant/Downloads/flowdiffusion-model/results/{datetime.now().day}-{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}/"
        args.checkpoint_path = "/media/doanthinhvo/OS/Users/doant/Downloads/flowdiffusion-model/checkpoints/"

    args.channels = 3
    args.image_size = 64
    args.device = "cuda"
    args.lr = 1e-4

    train(args)

if __name__ == '__main__':
    launch(debug=True)
# test FID score on pretrained model
from utils import *
from calculate_fid_score import *
from modules import Unet
from torch import optim
from ddpm import Diffusion

checkpoint_path = '/kaggle/input/deletesoon/flowdiffusion-model/checkpoints/baymax.pt'

dl = get_train_dataloader("/kaggle/input/deletesoon/flowdiffusion-model/data", 16)
dataloader = cycle(dl)
data = next(dataloader)
batch,_ = data
real_samples = batch.to('cuda')

model = Unet(
        dim=64,
        channels=3,
        dim_mults=(1, 2, 4, 8) 
    ).to('cuda')
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler=None

model, optimizer, scheduler, args = load_checkpoint_local(model,optimizer, scheduler, checkpoint_path)

diffusion = Diffusion(timesteps=1000, img_size=64, device='cuda')
all_images_list = diffusion.sample(model, 64, batch_size=32, channels=3)
all_images = all_images_list[-1]

fid_score = FIDScore('cuda')
fake_images = torch.from_numpy(all_images).to('cuda')
print(f"FID score: {fid_score.fid_score(real_samples, fake_images)}")



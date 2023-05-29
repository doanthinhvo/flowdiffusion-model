from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from utils import *
from einops import rearrange

class FIDScore:
    def __init__(self, device):
        self.inception_block_idx = 2048
        self.channels = 3 # remove
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx])
        self.device = device
        self.inception_v3.to(self.device)

    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)

        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...').cpu().numpy()

        mu = np.mean(features, axis = 0)
        sigma = np.cov(features, rowvar = False)
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):

        if self.channels == 1:
            real_samples, fake_samples = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (real_samples, fake_samples))

        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

if __name__ == '__main__':
    fid_score = FIDScore(device='cuda')
    # real_samples = torch.randn(100, 3, 64, 64).to('cuda')
    # fake_samples = torch.randn(100, 3, 64, 64).to('cuda')
    train_dataloader = get_train_dataloader("/media/doanthinhvo/OS/Users/doant/Downloads/flowdiffusion-model/data/", 16)
    train_dataloader = (cycle(train_dataloader))
    val_dataloader = get_val_dataloader("/media/doanthinhvo/OS/Users/doant/Downloads/flowdiffusion-model/data/", 16)
    val_dataloader = (cycle(val_dataloader))
    train_batch = next(train_dataloader)[0]
    val_batch = next(val_dataloader)[0]

    print(fid_score.fid_score(train_batch.to('cuda'), val_batch.to('cuda')))

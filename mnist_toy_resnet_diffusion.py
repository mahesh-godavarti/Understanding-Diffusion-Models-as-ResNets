# mnist_stepwise_resnet_denoiser.py
import math, os, argparse, random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    data_dir: str = "./data"
    out_dir: str  = "./runs/stepres_denoise"
    batch_size: int = 256
    epochs: int = 20
    lr: float = 2e-3
    weight_decay: float = 0.0
    num_workers: int = 4
    steps: int = 32            # N: total repeated layers
    base_ch: int = 64
    noise_embeddings: bool = False  # add via --noise_embeddings
    log_every: int = 100
    samples_per_epoch: int = 100
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    import numpy as np
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_neg_one_to_one(x):  # [0,1] -> [-1,1]
    return x * 2.0 - 1.0

def to_zero_one(x):        # [-1,1] -> [0,1]
    return x.add(1).div(2).clamp(0, 1)


# ----------------------------
# Data
# ----------------------------
def get_loader(cfg: Config):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_neg_one_to_one)
    ])
    ds = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    return loader

@torch.no_grad()
def get_test_batch(cfg: Config, count: int):
    """Grab a batch of real digits from the MNIST test set in [-1,1]."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_neg_one_to_one)
    ])
    test_ds = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=tfm)
    idx = torch.randperm(len(test_ds))[:count]
    xs = torch.stack([test_ds[i][0] for i in idx], dim=0)   # [B,1,28,28]
    return xs.to(cfg.device)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.gn1   = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.gn2   = nn.GroupNorm(8, out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        h = F.relu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.relu(h + self.skip(x))

class StepwiseResNetDenoiser(nn.Module):
    def __init__(self, ch=64, noise_embeddings: bool = False):
        super().__init__()
        self.noise_embeddings = noise_embeddings
        # Down
        self.enc1 = ResBlock(2 if self.noise_embeddings else 1, ch)          # x -> ch
        self.down = nn.Conv2d(ch, ch*2, 4, 2, 1)   # 28->14
        self.enc2 = ResBlock(ch*2, ch*2)
        # Up
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")         # 14->28
        self.dec1 = ResBlock(ch*3, ch)    # will concat skip
        self.outc = nn.Conv2d(ch, 1, 3, 1, 1)  # velocity field (no activation)

    def block(self, x, scale=None):
        if self.noise_embeddings:
            assert scale is not None, "scale required when --noise_embeddings is set"
            h0 = scale * torch.randn_like(x)
            x_in = torch.cat([x, h0], dim=1)
        else:
            x_in = x
        h1 = self.enc1(x_in)                   # [B,ch,28,28]
        h2 = self.down(h1)                   # [B,2ch,14,14]
        h3 = self.enc2(h2)                   # [B,2ch,14,14]
        hu = self.up(h3)                     # [B,2ch,28,28]
        h  = torch.cat([hu, h1], dim=1)      # skip
        h  = self.dec1(h)                    # [B,ch,28,28]
        v  = self.outc(h)                    # [B,1,28,28]
        return v

    def forward(self, x, scale):
        return x + scale * self.block(x)


    def repeat_block(self, x, k: int, steps: int):
        for i in range(max(1, k)):
            x = x + 1 / float(steps) * (self.block(x, (k-i)/float(steps)) if self.noise_embeddings else self.block(x))

        return x

    def forward_single_k(self, x, k: int, steps: int):
        x = x + k / float(steps) * (self.block(x, k/float(steps)) if self.noise_embeddings else self.block(x))

        return x

    def forward_k(self, x_k, k: int, steps: int):
        return self.repeat_block(x_k, k, steps)

    def forward_single_layer_k(self, x_k, k:int, steps: int):
        return x_k + 1 / float(steps) * (self.block(x_k, k/float(steps)) if self.noise_embeddings else self.block(x_k))

# ----------------------------
# Training loop
# ----------------------------
def train(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = cfg.device

    loader = get_loader(cfg)
    net = StepwiseResNetDenoiser(ch=cfg.base_ch, noise_embeddings=cfg.noise_embeddings).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)

    it = 0
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        for i, (x_clean, _) in enumerate(loader):
            x_clean = x_clean.to(device)  # [-1,1], target x0
            B = x_clean.size(0)

            # sample k in {1..N}
            k  = torch.randint(1, cfg.steps, (B, ), device=device)                     # [0,1)
            k = k.view(B, 1, 1, 1)
            t_k = k / float(cfg.steps)
            t_kminus1 = (k - 1) / float(cfg.steps)
            noise = torch.randn_like(x_clean)
            x_k = (1.0 - t_k) * x_clean + t_k * noise
            x_target = (1.0 - t_kminus1) * x_clean + t_kminus1 * noise

            # predict clean from x_k by unrolling EXACTLY k times
            #pred = net.forward_k(x_k, k, cfg.steps)
            # predict clean directly from x_k
            pred = net.forward_single_layer_k(x_k, k, cfg.steps)
            loss = F.mse_loss(cfg.steps*pred, cfg.steps*x_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            it += 1
            if (i + 1) % cfg.log_every == 0:
                print(f"Epoch {epoch:03d} | Iter {i+1:04d}/{len(loader)} | fm_loss {loss.item():.4f}")


        # samples each epoch
        sample_grids(net, cfg, epoch)

        # checkpoint
        torch.save({"model": net.state_dict(), "cfg": cfg.__dict__, "epoch": epoch},
                   f"{cfg.out_dir}/ckpt_{epoch:03d}.pt")

    print("Training complete.")


# ----------------------------
# Sampling:
#   Save three BEFORE|AFTER grids per epoch:
#   (A) Pure noise -> run N repeats
#   (B) 1 step away (t=1/N)  -> run LAST 1 layer
#   (C) 5 steps away (t=5/N) -> run LAST 5 layers
# ----------------------------
@torch.no_grad()
def sample_grids(net: StepwiseResNetDenoiser, cfg: Config, epoch: int):
    net.eval()
    B = cfg.samples_per_epoch
    device = cfg.device

    # Helper to stack before|after horizontally
    def side_by_side(before_img, after_img):
        return torch.cat([before_img, after_img], dim=2)  # concat along width

    # (A) Pure noise
    x_noise = torch.randn(B, 1, 28, 28, device=device)     # before (unbounded)
    y_noise = net.forward_k(x_noise, cfg.steps, cfg.steps)                      # after
    # For visualization of BEFORE, squash noise via tanh so it's in [-1,1]
    grid_before = vutils.make_grid(to_zero_one(torch.tanh(x_noise).cpu()),
                                   nrow=int(math.sqrt(B)), padding=2, normalize=False)
    grid_after  = vutils.make_grid(to_zero_one(y_noise.cpu()),
                                   nrow=int(math.sqrt(B)), padding=2, normalize=False)
    grid_pair   = side_by_side(grid_before, grid_after)
    vutils.save_image(grid_pair, f"{cfg.out_dir}/samples_epoch_{epoch:03d}_A_pure_noise_before_after.png")

    # Get real digits to form "k steps away" starts
    x_real = get_test_batch(cfg, B)                         # [-1,1]
    # (B) 1 step away: t = 1/N, run LAST 1 layer
    k1 = 1
    t1 = k1 / float(cfg.steps)
    x_k1 = (1.0 - t1) * x_real + t1 * torch.randn_like(x_real)
    y_k1 = net.forward_k(x_k1, k1, cfg.steps)                          # AFTER using last 1 layer (or 1 repeat)
    grid_before = vutils.make_grid(to_zero_one(x_k1.cpu()),
                                   nrow=int(math.sqrt(B)), padding=2, normalize=False)
    grid_after  = vutils.make_grid(to_zero_one(y_k1.cpu()),
                                   nrow=int(math.sqrt(B)), padding=2, normalize=False)
    grid_pair   = side_by_side(grid_before, grid_after)
    vutils.save_image(grid_pair, f"{cfg.out_dir}/samples_epoch_{epoch:03d}_B_k1_before_after.png")

    # (C) 5 steps away: t = 5/N, run LAST 5 layers
    k5 = min(5, cfg.steps)
    t5 = k5 / float(cfg.steps)
    x_k5 = (1.0 - t5) * x_real + t5 * torch.randn_like(x_real)
    y_k5 = net.forward_k(x_k5, k5, cfg.steps)
    grid_before = vutils.make_grid(to_zero_one(x_k5.cpu()),
                                   nrow=int(math.sqrt(B)), padding=2, normalize=False)
    grid_after  = vutils.make_grid(to_zero_one(y_k5.cpu()),
                                   nrow=int(math.sqrt(B)), padding=2, normalize=False)
    grid_pair   = side_by_side(grid_before, grid_after)
    vutils.save_image(grid_pair, f"{cfg.out_dir}/samples_epoch_{epoch:03d}_C_k5_before_after.png")


# ----------------------------
# CLI
# ----------------------------
def str2bool(s: str) -> bool:
    return str(s).lower() in ("1","true","yes","y","t")

def main():
    p = argparse.ArgumentParser("Step-wise ResNet MNIST Denoiser + side-by-side sample grids")
    p.add_argument("--data_dir", type=str, default=Config.data_dir)
    p.add_argument("--out_dir", type=str, default=Config.out_dir)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--base_ch", type=int, default=Config.base_ch)
    p.add_argument("--noise_embeddings", action="store_true")
    p.add_argument("--log_every", type=int, default=Config.log_every)
    p.add_argument("--samples_per_epoch", type=int, default=Config.samples_per_epoch)
    p.add_argument("--seed", type=int, default=Config.seed)
    args = p.parse_args()

    cfg = Config(
        data_dir=args.data_dir, out_dir=args.out_dir, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        num_workers=args.num_workers, steps=args.steps, base_ch=args.base_ch,
        log_every=args.log_every,
        samples_per_epoch=args.samples_per_epoch, seed=args.seed, noise_embeddings=bool(args.noise_embeddings),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    train(cfg)

if __name__ == "__main__":
    main()

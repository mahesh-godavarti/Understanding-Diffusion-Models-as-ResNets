#!/usr/bin/env python3
import os, math, argparse, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils

# ----------------------------
# Utilities
# ----------------------------
def to_zero_one(x):        # [-1,1] -> [0,1]
    return x.add(1).div(2).clamp(0, 1)

def seed_all(seed=42):
    import random, numpy as np
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Model (noise-embedding variant, matching your training file)
# ----------------------------
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
    """
    Noise-embedding version:
      - enc1 takes 2 channels: x and a scaled Gaussian channel (k/N * N(0,1))
      - forward_* apply residual updates with 1/steps scale factors
    """
    def __init__(self, ch=64, use_noise_emb=True):
        super().__init__()
        self.use_noise_emb = use_noise_emb
        # Down
        self.enc1 = ResBlock(2 if self.use_noise_emb else 1, ch)          # x + optional noise_embed -> ch
        self.down = nn.Conv2d(ch, ch*2, 4, 2, 1)   # 28->14
        self.enc2 = ResBlock(ch*2, ch*2)
        # Up
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")         # 14->28
        self.dec1 = ResBlock(ch*3, ch)    # concat skip
        self.outc = nn.Conv2d(ch, 1, 3, 1, 1)  # residual/velocity field (no activation)

    def block(self, x, scale):
        # scale encodes timestep: scale ~ k/N
        if self.use_noise_emb:
            h0 = scale * torch.randn_like(x)
            h1 = self.enc1(torch.cat([x, h0], dim=1))           # [B,ch,28,28]
        else:
            h1 = self.enc1(x)                                   # [B,ch,28,28]
        h2 = self.down(h1)                                  # [B,2ch,14,14]
        h3 = self.enc2(h2)                                  # [B,2ch,14,14]
        hu = self.up(h3)                                    # [B,2ch,28,28]
        h  = torch.cat([hu, h1], dim=1)                     # skip concat
        h  = self.dec1(h)                                   # [B,ch,28,28]
        v  = self.outc(h)                                   # [B,1,28,28]
        return v

    def forward(self, x, scale):
        return x + scale * self.block(x, scale)

    def forward_single_layer_k(self, x_k, k: int, steps: int):
        # EXACTLY matches your training/sampling definition:
        # x_{k-1} = x_k + (1/N) * block(x_k, k/N)
        return x_k + 1.0/float(steps) * self.block(x_k, k/float(steps))

# ----------------------------
# Grid helpers
# ----------------------------
def make_grid(x, nrow):
    return vutils.make_grid(to_zero_one(x.cpu()), nrow=nrow, padding=2, normalize=False)

def save_grid(x, path, nrow):
    grid = make_grid(x, nrow)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(grid, path)

# ----------------------------
# Main: load ckpt, run from pure noise, dump a grid for *each* step
# ----------------------------
def main():
    p = argparse.ArgumentParser("Generate per-step grids that EXACTLY match original sampling schedule")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt saved during training")
    p.add_argument("--steps", type=int, default=None, help="Total steps N (if None, try to infer from ckpt cfg)")
    p.add_argument("--samples", type=int, default=100, help="Number of images per grid (e.g., 100 -> 10x10)")
    p.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    p.add_argument("--noise_embeddings", action="store_true")
    p.add_argument("--ch", type=int, default=64, help="Base channels (should match training)")
    p.add_argument("--out_dir", type=str, default="./grids_per_step", help="Output directory")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)  # support raw state_dict or wrapped
    cfg_ckpt = ckpt.get("cfg", {})

    steps = int(args.steps or cfg_ckpt.get("steps", 32))
    ch    = int(args.ch or cfg_ckpt.get("base_ch", 64))

    # Build model and load weights
    net = StepwiseResNetDenoiser(ch=ch, use_noise_emb=args.noise_embeddings).to(device)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[warn] load_state_dict: missing:", missing, "unexpected:", unexpected)

    net.eval()

    # Start from pure noise (unbounded), like your sampling
    B = args.samples
    nrow = int(math.sqrt(B)) if int(math.sqrt(B))**2 == B else min(10, B)
    x0 = torch.randn(B, 1, 28, 28, device=device)

    # Save grid for step 0 (i.e., pure noise squashed to [-1,1] via tanh for visibility)
    save_grid(torch.tanh(x0.detach()), os.path.join(args.out_dir, f"step_{0:03d}.png"), nrow=nrow)

    # In-place evolution with the *decreasing* schedule: k = N, N-1, ..., 1
    x_working = x0.clone()
    for s in range(1, steps + 1):
        k_current = steps - s + 1        # first step uses k=N (scale=1), then N-1, ..., 1
        with torch.no_grad():
            x_working = net.forward_single_layer_k(x_working, k_current, steps)
        save_grid(x_working.detach(), os.path.join(args.out_dir, f"step_{s:03d}.png"), nrow=nrow)

    # Optional: also write the final grid with a name matching original full unroll (sanity check)
    save_grid(x_working.detach(), os.path.join(args.out_dir, f"final_full_unroll.png"), nrow=nrow)

    print(f"[done] Wrote per-step grids to: {args.out_dir}\n"
          f"Tip: step_{steps:03d}.png should match the original net.forward_k(x_noise, N, N) output.")

if __name__ == "__main__":
    main()

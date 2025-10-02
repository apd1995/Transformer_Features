#!/usr/bin/env python3
# vit_cifar_ddp.py — variable # of blocks, 1-GPU or multi-GPU (DDP) ready.
import argparse, os, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# ----- Model -----
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                 # (B,D,H',W')
        return x.flatten(2).transpose(1, 2)  # (B,N,D)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(mlp_ratio * embed_dim)
        self.mlp  = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=768, mlp_ratio=4.0,
                 num_heads=1, num_blocks=1, num_classes=10):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.patch.num_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio)
                                     for _ in range(num_blocks)])
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, 0])

# ----- Data -----
def cifar10_loaders(bs=128, workers=0, distributed=False):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR10(root="./data", train=True,  download=True, transform=tfm)
    test  = datasets.CIFAR10(root="./data", train=False, download=True, transform= tfm)
    train_samp = DistributedSampler(train, shuffle=True) if distributed else None
    test_samp  = DistributedSampler(test,  shuffle=False) if distributed else None
    train_loader = DataLoader(train, batch_size=bs, shuffle=(train_samp is None),
                              sampler=train_samp, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=bs, shuffle=False,
                              sampler=test_samp,  num_workers=workers, pin_memory=True)
    return train_loader, test_loader, train_samp, test_samp

@torch.no_grad()
def eval_top1(model, loader, device):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return 100.0 * correct / total

def is_dist():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=0)        # -c 1 ⇒ use 0
    ap.add_argument("--num_heads", type=int, default=1)
    ap.add_argument("--num_blocks", type=int, default=1)     # NEW: variable depth
    args = ap.parse_args()

    distributed = is_dist()
    print(f"Distributed: {distributed}")
    
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (not distributed) and device.type == "cuda":
        print(f"✅ Using CUDA on {torch.cuda.get_device_name(0)}")
    elif distributed:
        if dist.get_rank() == 0:
            print(f"✅ DDP with {dist.get_world_size()} GPUs")
    else:
        print("⚠️ CUDA not available, using CPU")

    train_loader, test_loader, train_samp, test_samp = cifar10_loaders(
        bs=args.bs, workers=args.workers, distributed=distributed
    )

    model = ViT(embed_dim=768, mlp_ratio=args.mlp_ratio,
                num_heads=args.num_heads, num_blocks=args.num_blocks,
                num_classes=10).to(device)

    if distributed:
        model = DDP(model, device_ids=[device.index])

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        if distributed and train_samp is not None:
            train_samp.set_epoch(ep)
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()

        # eval (rank 0 only for neat logs)
        if (not distributed) or dist.get_rank() == 0:
            top1 = eval_top1(model.module if distributed else model, test_loader, device)
            print(f"Epoch {ep+1}/{args.epochs} | CIFAR-10 test top-1: {top1:.2f}%")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

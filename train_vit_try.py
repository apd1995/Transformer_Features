#!/usr/bin/env python3
# vit_oneblock_cifar10.py
# ViT-B style (D=768, MLP ratio default 4) but single block & 1 head, for CIFAR-10 (32x32).
import argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----- Model -----
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):                # (B,C,H,W)->(B,N,D)
        x = self.proj(x)                 # (B,D,H',W')
        return x.flatten(2).transpose(1, 2)

class OneBlockViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=768, mlp_ratio=4, num_heads=1, num_classes=10):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.patch.num_patches, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(mlp_ratio * embed_dim)
        self.mlp  = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)                           # (B,N,D)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1) + self.pos
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return self.head(x[:, 0])                   # CLS → logits

# ----- Data -----
def cifar10_loaders(bs=128, workers=2):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR10(root="./data", train=True,  download=True, transform=tfm)
    test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train, batch_size=bs, shuffle=True,  num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader

@torch.no_grad()
def eval_top1(model, loader, device):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return 100.0 * correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available device: {device}")
    train_loader, test_loader = cifar10_loaders(args.bs, args.workers)
    model = OneBlockViT(embed_dim=768, mlp_ratio=args.mlp_ratio, num_heads=args.num_heads, num_classes=10).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        top1 = eval_top1(model, test_loader, device)
        print(f"Epoch {ep+1}/{args.epochs} | CIFAR-10 test top-1: {top1:.2f}%")

if __name__ == "__main__":
    main()


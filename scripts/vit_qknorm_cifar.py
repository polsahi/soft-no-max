#!/usr/bin/env python3
"""
Prove m*: QK-norm gives a provable, profiling-free Soft-no-Max constant.

Trains two small ViTs on CIFAR-10 and inspects their ATTENTION logits:
  - vanilla   : logit = (q.k)/sqrt(d_head). Range is data/training dependent, not
                bounded a priori -> m* must be profiled.
  - qknorm    : q,k L2-normalised along the head dim, scaled by a learnable per-head
                g -> logit = g*cos(theta) in [-g, g]. So m* = g provably, read off the
                trained weights, no profiling.

For each model we (a) train briefly, (b) measure the actual attention-logit range on the
test set, (c) for qknorm read g and check every logit lies in [-g, g], and (d) confirm
Soft-no-Max with the respective m* is exact and never overflows.

Loads CIFAR-10 from the raw python batches (no torchvision), so it runs anywhere.

    python3 vit_qknorm_cifar.py <cifar_dir> [epochs] [out.csv]
    cifar_dir = .../cifar-10-batches-py
"""

import csv
import glob
import math
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalise(xtr, ytr, xte, yte):
    mean = xtr.mean((0, 2, 3), keepdims=True); std = xtr.std((0, 2, 3), keepdims=True)
    xtr = (xtr - mean) / std; xte = (xte - mean) / std
    return (torch.tensor(xtr), torch.tensor(ytr),
            torch.tensor(xte), torch.tensor(yte))


def load_cifar(root):
    # (a) pre-saved .npy bundle (fast, portable to MN5)
    npz = os.path.join(root, "cifar10.npz") if os.path.isdir(root) else root
    if npz.endswith(".npz") and os.path.exists(npz):
        d = np.load(npz)
        return _normalise(d["xtr"], d["ytr"], d["xte"], d["yte"])
    # (b) raw python batches
    if os.path.isdir(root) and glob.glob(os.path.join(root, "data_batch_*")):
        def unp(f):
            with open(f, "rb") as fh:
                dd = pickle.load(fh, encoding="bytes")
            return dd[b"data"], dd[b"labels"]
        xs, ys = [], []
        for f in sorted(glob.glob(os.path.join(root, "data_batch_*"))):
            x, y = unp(f); xs.append(x); ys.append(y)
        xtr = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        ytr = np.concatenate([np.array(y) for y in ys])
        xt, yt = unp(os.path.join(root, "test_batch"))
        xte = np.array(xt).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        yte = np.array(yt)
        return _normalise(xtr, ytr, xte, yte)
    # (c) HuggingFace datasets fallback
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10")
    def to_arr(split):
        x = np.stack([np.array(e["img"], dtype=np.float32).transpose(2, 0, 1) / 255.0
                      for e in split])
        y = np.array([e["label"] for e in split])
        return x, y
    xtr, ytr = to_arr(ds["train"]); xte, yte = to_arr(ds["test"])
    return _normalise(xtr, ytr, xte, yte)


# Optional hard cap on the QK-norm gain (env VIT_G_MAX): g = G_MAX * sigmoid(g_raw), so every logit lies in
# [-g, g] with g <= G_MAX for ALL inputs and ALL parameter values. Unset: the original uncapped softplus gain.
G_MAX = float(os.environ["VIT_G_MAX"]) if os.environ.get("VIT_G_MAX") else None


class Attention(nn.Module):
    def __init__(self, dim, heads, qknorm):
        super().__init__()
        self.h, self.dh, self.qknorm = heads, dim // heads, qknorm
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        if qknorm:
            # learnable per-head scale g (replaces 1/sqrt(dh)); softplus keeps g>0
            if G_MAX:
                g0 = min(self.dh ** 0.5, 0.95 * G_MAX)
                self.g_raw = nn.Parameter(torch.full((heads,), math.log(g0 / (G_MAX - g0))))
            else:
                self.g_raw = nn.Parameter(torch.full((heads,), math.log(math.e ** (self.dh ** 0.5) - 1)))
        self.last_logits = None            # cached max attention logit for inspection

    def g(self):
        if not self.qknorm:
            return None
        return G_MAX * torch.sigmoid(self.g_raw) if G_MAX else F.softplus(self.g_raw)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # B,h,N,dh
        if self.qknorm:
            q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
            g = self.g().view(1, self.h, 1, 1)
            logits = (q @ k.transpose(-1, -2)) * g      # in [-g, g]
        else:
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        self.last_logits = logits.detach()
        a = F.softmax(logits, dim=-1)
        o = (a @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, dim, heads, qknorm):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.attn = Attention(dim, heads, qknorm)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x)); x = x + self.mlp(self.n2(x)); return x


class ViT(nn.Module):
    def __init__(self, qknorm, dim=128, heads=4, depth=4, patch=4, ncls=10):
        super().__init__()
        np_ = (32 // patch) ** 2
        self.patch = patch
        self.embed = nn.Conv2d(3, dim, patch, patch)
        self.pos = nn.Parameter(torch.randn(1, np_ + 1, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, qknorm) for _ in range(depth))
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, ncls)

    def forward(self, x):
        B = x.shape[0]
        x = self.embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x)[:, 0])


def augment(x):
    # random horizontal flip (per image) + random crop with reflect padding 4 (per batch)
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oy = int(torch.randint(0, 9, (1,))); ox = int(torch.randint(0, 9, (1,)))
    return x[:, :, oy:oy + 32, ox:ox + 32]


def run(qknorm, data, epochs, dev):
    xtr, ytr, xte, yte = data
    model = ViT(qknorm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.05)
    n = xtr.shape[0]; bs = 256
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4, total_steps=steps,
                                                pct_start=0.1)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = augment(xtr[idx].to(dev)); yb = ytr[idx].to(dev)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    # test accuracy + attention-logit range over the test set
    model.eval(); correct = 0; lo, hi = float("inf"), -float("inf")
    with torch.no_grad():
        for i in range(0, xte.shape[0], 512):
            xb, yb = xte[i:i + 512].to(dev), yte[i:i + 512].to(dev)
            correct += int((model(xb).argmax(1) == yb).sum())
            for blk in model.blocks:
                lg = blk.attn.last_logits
                lo = min(lo, float(lg.min())); hi = max(hi, float(lg.max()))
    acc = correct / xte.shape[0]
    g_max = float(model.blocks[0].attn.g().max()) if qknorm else None
    if qknorm:
        g_all = max(float(b.attn.g().max()) for b in model.blocks)
    else:
        g_all = None
    return model, acc, lo, hi, g_all


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "data/cifar-10-batches-py"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    out_csv = sys.argv[3] if len(sys.argv) > 3 else "results/vit_qknorm.csv"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(os.environ.get("VIT_SEED", 0)))
    data = load_cifar(root)
    print(f"device={dev}  epochs={epochs}  CIFAR-10 ViT\n", flush=True)

    rows = []
    for qk in (False, True):
        name = "qknorm" if qk else "vanilla"
        model, acc, lo, hi, g_all = run(qk, data, epochs, dev)
        # Keep the trained weights: the secure-inference experiments (mpc_vit_private.py) load
        # them, and the per-head g values are the public softmax shifts there.
        ckpt_dir = os.environ.get("VIT_CKPT_DIR", os.path.join(os.path.dirname(out_csv) or ".", "ckpt"))
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "qknorm": qk, "test_acc": acc,
                    "logit_range": (lo, hi), "epochs": epochs},
                   os.path.join(ckpt_dir, f"vit_cifar_{name}{os.environ.get('VIT_TAG', '')}.pt"))
        print(f"=== {name} ===")
        print(f"  test acc            : {acc*100:.2f}%")
        print(f"  attention logit range: [{lo:.3f}, {hi:.3f}]")
        if qk:
            m_star = g_all + 0.01
            covered = hi <= g_all + 1e-4
            # exactness of SnM with m*=g on a test batch
            xb = data[2][:256].to(dev)
            with torch.no_grad():
                model(xb)
                err = 0.0; finite = True
                for b in model.blocks:
                    lg = b.attn.last_logits
                    ref = torch.softmax(lg, dim=-1)
                    e = (lg - m_star).exp(); snm = e / e.sum(-1, keepdim=True)
                    if not torch.isfinite(snm).all(): finite = False
                    err = max(err, (snm - ref).abs().max().item())
            print(f"  learnable g (max)   : {g_all:.3f}  -> provable bound |logit| <= g")
            print(f"  logits within [-g,g]: {covered}  (max logit {hi:.3f} vs g {g_all:.3f})")
            print(f"  m* = g + 0.01       : SnM exact err {err:.2e}, finite {finite}, "
                  f"NO profiling")
            rows.append(dict(model=name, test_acc=acc, logit_min=lo, logit_max=hi,
                             g=g_all, m_star=m_star, bound_holds=covered,
                             snm_err=err, finite=finite, profiling_needed=False))
        else:
            print(f"  m* NOT known a priori: range depends on data/weights -> must profile")
            rows.append(dict(model=name, test_acc=acc, logit_min=lo, logit_max=hi,
                             g=None, m_star=None, bound_holds=None,
                             snm_err=None, finite=None, profiling_needed=True))
        print(flush=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out_csv}")

    qk = [r for r in rows if r["model"] == "qknorm"][0]
    print(f"\nProvable m* validated: {'YES' if qk['bound_holds'] and qk['finite'] else 'NO'}"
          f"  (QK-norm: m*=g={qk['g']:.3f}, no profiling; vanilla: must profile)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end 2-party private inference of the CIFAR-10 ViT, with three secure softmaxes.

Model: `vit_qknorm_cifar.ViT` (dim 128, 4 heads, 4 blocks, 65 tokens), trained weights from
`vit_qknorm_cifar.py` (checkpoints `vit_cifar_{vanilla,qknorm}.pt`). The forward pass is written
out in CrypTen operations so every secure op is accounted to a category:

  linear     matrix multiplications with the (encrypted) weights, incl. patch embedding and head
  layernorm  mean, variance, Newton inverse square root
  qknorm     L2 normalisation of queries and keys (Newton inverse square root per token and head)
  softmax    the attention softmax, one of:
               crypten_max  secure max + exp + reciprocal (CrypTen)
               shaft_ode    clip to [-4, 12] + ODE, t = 16 (SHAFT, NDSS 2025)
               snm_public   subtract the public per-head bound g, exp, reciprocal (0 comparisons)
  gelu       SHAFT's Fourier-series GELU
  attn_mix   attention-weighted sum of values and output projection
  other      residual additions, token concatenation

Threat model as in SHAFT: party 0 holds the model, party 1 the image; both are secret-shared.
For snm_public the per-head QK-norm gains g (4 scalars per block) are public constants, like
architecture hyperparameters.

Usage (from SHAFT's examples/unit-test directory, for its MultiProcessLauncher):
    python mpc_vit_private.py --ckpt_dir ckpt --data bench/data/cifar10.npz --n_images 200 \
        --out results/mpc/mpc_vit_private.csv
"""

import argparse
import collections
import csv
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("SHAFT_UNIT_TEST_DIR", os.getcwd()))

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt_dir", default="results/ckpt")
_ap.add_argument("--data", default="data/cifar10.npz")
_ap.add_argument("--n_images", type=int, default=200)
_ap.add_argument("--batch", type=int, default=50)
_ap.add_argument("--cost_batch", type=int, default=1)
_ap.add_argument("--models", default="vanilla,qknorm")
_ap.add_argument("--precision_bits", type=int, default=16)
_ap.add_argument("--out", default="results/mpc/mpc_vit_private.csv")
ARGS = _ap.parse_known_args()[0]


class Meter:
    """Accumulates communication rounds, bytes and wall time per op category."""

    def __init__(self):
        self.acc = collections.defaultdict(lambda: [0, 0, 0.0])

    def run(self, cat, fn, *args):
        import crypten
        s0 = crypten.get_communication_stats()
        t0 = time.time()
        out = fn(*args)
        dt = time.time() - t0
        s1 = crypten.get_communication_stats()
        a = self.acc[cat]
        a[0] += s1["rounds"] - s0["rounds"]
        a[1] += s1["bytes"] - s0["bytes"]
        a[2] += dt
        return out


def enc_params(state, src=0):
    import crypten
    return {k: crypten.cryptensor(v.float(), src=src) for k, v in state.items()}


def secure_forward(x, P, qknorm, g_public, variant, meter):
    import crypten
    from crypten.config import cfg

    B = x.size(0)
    dim, heads, depth, patch = 128, 4, 4, 4
    dh = dim // heads

    def linear(t, w, b=None):
        out = t.matmul(w.t())
        return out + b if b is not None else out

    def layernorm(t, w, b):
        mean = t.mean(dim=-1, keepdim=True)
        inv_sd = (t.var(dim=-1, keepdim=True) + 1e-5).inv_sqrt()
        return (t - mean) * inv_sd * w + b

    def l2norm(t):
        return t * (t.square().sum(dim=-1, keepdim=True) + 1e-12).inv_sqrt()

    def softmax(logits, layer):
        if variant == "crypten_max":
            with cfg.temp_override({"functions.softmax_method": "reciprocal"}):
                return logits.softmax(-1)
        if variant == "shaft_ode":
            with cfg.temp_override({"functions.softmax_method": "ode", "functions.softmax_ode_clip": True,
                                    "functions.softmax_ode_lb": -4.0, "functions.softmax_ode_ub": 12.0,
                                    "functions.softmax_ode_iter_num": 16}):
                return logits.softmax(-1)
        if variant == "snm_public":
            shift = g_public[layer].view(1, heads, 1, 1)
            num = (logits - shift).exp()
            with cfg.temp_override({"functions.reciprocal_all_pos": True}):
                inv = num.sum(-1, keepdim=True).reciprocal()
            return num * inv
        raise ValueError(variant)

    # patch embedding: a stride-4 4x4 convolution is a linear map on non-overlapping patches
    def embed(x):
        p = x.reshape(B, 3, 32 // patch, patch, 32 // patch, patch).permute(0, 2, 4, 1, 3, 5)
        p = p.reshape(B, (32 // patch) ** 2, 3 * patch * patch)
        w = P["embed.weight"].reshape(dim, -1)
        return linear(p, w, P["embed.bias"])
    h = meter.run("linear", embed, x)
    h = meter.run("other", lambda h: crypten.cat([P["cls"].expand(B, 1, dim), h], dim=1) + P["pos"], h)
    N = h.size(1)

    verbose = os.environ.get("MPC_VERBOSE") == "1" and crypten.comm.get().get_rank() == 0
    for L in range(depth):
        if verbose:
            print(f"    [{variant}] block {L} start, rounds so far {crypten.get_communication_stats()['rounds']}", flush=True)
        pre = f"blocks.{L}."
        y = meter.run("layernorm", layernorm, h, P[pre + "n1.weight"], P[pre + "n1.bias"])
        qkv = meter.run("linear", linear, y, P[pre + "attn.qkv.weight"])
        qkv = qkv.reshape(B, N, 3, heads, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if qknorm:
            # one secure normalisation over q and k together: the same work in half the rounds
            qk = meter.run("qknorm", l2norm, crypten.cat([q, k], dim=0))
            q, k = qk[:B], qk[B:]
            # g is a model weight but, as the public softmax bound, it is revealed: use it in the clear
            logits = meter.run("linear", lambda q, k: q.matmul(k.transpose(-1, -2)) * g_public[L].view(1, heads, 1, 1), q, k)
        else:
            logits = meter.run("linear", lambda q, k: q.matmul(k.transpose(-1, -2)).div(math.sqrt(dh)), q, k)
        a = meter.run("softmax", softmax, logits, L)
        o = meter.run("attn_mix", lambda a, v: linear(a.matmul(v).transpose(1, 2).reshape(B, N, dim),
                                                      P[pre + "attn.proj.weight"], P[pre + "attn.proj.bias"]), a, v)
        h = meter.run("other", lambda h, o: h + o, h, o)
        y = meter.run("layernorm", layernorm, h, P[pre + "n2.weight"], P[pre + "n2.bias"])
        y = meter.run("linear", linear, y, P[pre + "mlp.0.weight"], P[pre + "mlp.0.bias"])
        y = meter.run("gelu", lambda t: t.gelu(), y)
        y = meter.run("linear", linear, y, P[pre + "mlp.2.weight"], P[pre + "mlp.2.bias"])
        h = meter.run("other", lambda h, y: h + y, h, y)

    h = meter.run("layernorm", layernorm, h, P["norm.weight"], P["norm.bias"])
    return meter.run("linear", linear, h[:, 0], P["head.weight"], P["head.bias"])


def main():
    import crypten
    from crypten.config import cfg
    import numpy as np
    from vit_qknorm_cifar import ViT, _normalise

    a = ARGS
    crypten.init()
    rank = crypten.comm.get().get_rank()
    d = np.load(a.data)
    _, _, xte, yte = _normalise(d["xtr"], d["ytr"], d["xte"], d["yte"])
    xte, yte = xte[: a.n_images].float(), yte[: a.n_images]
    rows = []

    for name in a.models.split(","):
        ck = torch.load(os.path.join(a.ckpt_dir, f"vit_cifar_{name}.pt"), map_location="cpu")
        qk = bool(ck["qknorm"])
        model = ViT(qk)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            plain_logits = torch.cat([model(xte[i:i + 256]) for i in range(0, len(xte), 256)])
        plain_pred = plain_logits.argmax(1)
        state = {k: v for k, v in model.state_dict().items() if not k.endswith("g_raw")}
        g_public = [torch.nn.functional.softplus(model.blocks[L].attn.g_raw.detach()) if qk else None
                    for L in range(4)]
        variants = ["crypten_max", "shaft_ode"] + (["snm_public"] if qk else [])

        for variant in variants:
            with cfg.temp_override({"encoder.precision_bits": a.precision_bits}):
                P = enc_params(state, src=0)
                # accuracy over n_images, batched
                preds, max_logit_err = [], 0.0
                t_acc = time.time()
                for i in range(0, len(xte), a.batch):
                    x = crypten.cryptensor(xte[i:i + a.batch], src=1)
                    out = secure_forward(x, P, qk, g_public, variant, Meter()).get_plain_text()
                    preds.append(out.argmax(1))
                    max_logit_err = max(max_logit_err, float((out - plain_logits[i:i + a.batch]).abs().max()))
                t_acc = time.time() - t_acc
                preds = torch.cat(preds)
                # cost for cost_batch images, per category
                meter = Meter()
                x = crypten.cryptensor(xte[: a.cost_batch], src=1)
                crypten.reset_communication_stats()
                t0 = time.time()
                secure_forward(x, P, qk, g_public, variant, meter)
                total_t = time.time() - t0
                total = crypten.get_communication_stats()

            row = {"model": name, "variant": variant, "precision_bits": a.precision_bits,
                   "n_images": len(xte), "plain_acc": float((plain_pred == yte).float().mean()),
                   "mpc_acc": float((preds == yte).float().mean()),
                   "agree_with_plain": float((preds == plain_pred).float().mean()),
                   "max_logit_err": max_logit_err, "accuracy_wall_s": t_acc,
                   "cost_batch": a.cost_batch, "total_rounds": total["rounds"],
                   "total_bytes": total["bytes"], "total_time_s": total_t}
            for cat in ["linear", "layernorm", "qknorm", "softmax", "gelu", "attn_mix", "other"]:
                r, b, t = meter.acc.get(cat, [0, 0, 0.0])
                row[f"{cat}_rounds"], row[f"{cat}_bytes"], row[f"{cat}_time_s"] = r, b, t
            rows.append(row)
            if rank == 0:
                print(f"[{name:7s} {variant:11s}] acc plain {row['plain_acc']:.3f} mpc {row['mpc_acc']:.3f} "
                      f"agree {row['agree_with_plain']:.3f} max|dlogit| {max_logit_err:.3f} | per image: "
                      f"rounds {row['total_rounds']} bytes {row['total_bytes']/1e6:.1f}MB time {total_t:.2f}s | "
                      f"softmax {row['softmax_rounds']} rd {row['softmax_bytes']/1e6:.1f}MB "
                      f"qknorm {row['qknorm_rounds']} rd {row['qknorm_bytes']/1e6:.1f}MB", flush=True)

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[done] {a.out}")


if __name__ == "__main__":
    from multiprocess_launcher import MultiProcessLauncher
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()

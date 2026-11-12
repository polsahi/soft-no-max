#!/usr/bin/env python3
"""Secure 2-party cost of transformer models, per op category and softmax variant.

Communication rounds and bytes in secret-shared inference do not depend on the weight values, so
models run with random encrypted weights on one encrypted input of --seq_len tokens. Blocks are
written out in CrypTen operations (SHAFT's fork) so every secure op is accounted to a category.

Architectures (--arch):
  bert       BERT post-LayerNorm encoder layers only (the embedding lookup is identical for every
             variant and left out)
  bert_full  the whole BERT classifier: private token ids one-hot against the encrypted embedding
             table, position/type embeddings and embedding LayerNorm, the encoder, pooler (tanh)
             and classifier
  gpt2       GPT-2-style pre-LayerNorm causal decoder (no LM head). The causal mask is applied as a
             public affine map that sends masked logits to -30 before the softmax, so no variant
             needs an additive -inf in fixed point

Softmax variants:
  vanilla + crypten_max   q.k/sqrt(d), secure max + exp + reciprocal (CrypTen)
  vanilla + shaft_ode     q.k/sqrt(d), clip to [-4, 12] + ODE, t = 16 (SHAFT, NDSS 2025)
  vanilla + crypten_max_rescale / shaft_ode_narrow / snm_shifted   fairness controls: CrypTen max with our public rescale;
                          SHAFT with clip [-6,0] and 8 steps (cost only); constant shift on a vanilla model, no QK-norm (cost only)
  qknorm  + snm_shifted   q, k L2-normalised (one secure inverse square root over q and k),
                          logits g*cos - g in [-2g, 0], exp + reciprocal, no comparison

Categories: embedding, linear, layernorm, qknorm, softmax, gelu, attn_mix, head, other.

Two ways to run:
  one machine   python mpc_encoder_cost.py ...           (both parties as local processes)
  two machines  SNM_MULTINODE=1 RANK=<0|1> WORLD_SIZE=2 RENDEZVOUS=tcp://<rank0 host>:<port> \
                python mpc_encoder_cost.py ... --net_probe
                (one party per machine, e.g. under srun; --net_probe measures the link first)

Usage (from SHAFT's examples/unit-test directory):
    python mpc_encoder_cost.py --arch bert --layers 12 --hidden 768 --heads 12 --seq_lens 128,256 \
        --out results/mpc/mpc_encoder_cost.csv
"""

import argparse
import collections
import csv
import math
import os
import socket
import sys
import time

import torch

sys.path.insert(0, os.environ.get("SHAFT_UNIT_TEST_DIR", os.getcwd()))

_ap = argparse.ArgumentParser()
_ap.add_argument("--arch", default="bert", choices=["bert", "bert_full", "gpt2"])
_ap.add_argument("--layers", type=int, default=12)
_ap.add_argument("--hidden", type=int, default=768)
_ap.add_argument("--heads", type=int, default=12)
_ap.add_argument("--vocab", type=int, default=30522)
_ap.add_argument("--seq_lens", default="128")
_ap.add_argument("--g", type=float, default=5.5)
_ap.add_argument("--configs", default="vanilla:crypten_max,vanilla:shaft_ode,qknorm:snm_shifted")
_ap.add_argument("--net_probe", action="store_true", help="measure bandwidth and round-trip latency between the parties first")
_ap.add_argument("--out", default="results/mpc/mpc_encoder_cost.csv")
ARGS = _ap.parse_known_args()[0]

CATS = ["embedding", "linear", "layernorm", "qknorm", "softmax", "gelu", "attn_mix", "head", "other"]
MASK_VALUE = -30.0


class Meter:
    def __init__(self):
        self.acc = collections.defaultdict(lambda: [0, 0, 0.0])

    def run(self, cat, fn, *args):
        import crypten
        s0 = crypten.get_communication_stats(); t0 = time.time()
        out = fn(*args)
        dt = time.time() - t0; s1 = crypten.get_communication_stats()
        a = self.acc[cat]
        a[0] += s1["rounds"] - s0["rounds"]; a[1] += s1["bytes"] - s0["bytes"]; a[2] += dt
        return out


def net_probe(rank):
    """Point-to-point bandwidth (32 MB float tensor, best of 5) and round-trip latency (200 pings)."""
    import torch.distributed as dist
    big = torch.randn(8_000_000)
    bw = []
    for _ in range(5):
        dist.barrier()
        t0 = time.time()
        if rank == 0:
            dist.send(big, dst=1)
        else:
            dist.recv(big, src=0)
        dist.barrier()
        bw.append(time.time() - t0)
    gbps = 8 * big.numel() * 4 / min(bw) / 1e9
    ping = torch.zeros(1)
    dist.barrier()
    t0 = time.time()
    for _ in range(200):
        if rank == 0:
            dist.send(ping, dst=1); dist.recv(ping, src=1)
        else:
            dist.recv(ping, src=0); dist.send(ping, dst=0)
    rtt_ms = (time.time() - t0) / 200 * 1e3
    return gbps, rtt_ms


def softmax(logits, method):
    from crypten.config import cfg
    n = logits.size(-1)
    c = max(1.0, n / 64.0)   # public rescale of the row sum, as given to our method
    if method == "crypten_max":
        with cfg.temp_override({"functions.softmax_method": "reciprocal"}):
            return logits.softmax(-1)
    if method == "crypten_max_rescale":
        # fairness control: CrypTen's secure-max softmax with the same public rescale as ours
        with cfg.temp_override({"functions.softmax_method": "reciprocal", "functions.softmax_shift_rescale": c}):
            return logits.softmax(-1)
    if method == "shaft_ode":
        with cfg.temp_override({"functions.softmax_method": "ode", "functions.softmax_ode_clip": True,
                                "functions.softmax_ode_lb": -4.0, "functions.softmax_ode_ub": 12.0,
                                "functions.softmax_ode_iter_num": 16}):
            return logits.softmax(-1)
    if method == "shaft_ode_narrow":
        # fairness control (cost only, accuracy not validated): SHAFT with the clip narrowed to the bounded range
        # [-6, 0] of a g<=3 model (width 6 <= t) and 8 steps instead of 16
        with cfg.temp_override({"functions.softmax_method": "ode", "functions.softmax_ode_clip": True,
                                "functions.softmax_ode_lb": -6.0, "functions.softmax_ode_ub": 0.0,
                                "functions.softmax_ode_iter_num": 8}):
            return logits.softmax(-1)
    if method == "snm_shifted":
        with cfg.temp_override({"functions.softmax_method": "shifted", "functions.softmax_shift_rescale": c}):
            return logits.softmax(-1)
    raise ValueError(method)


def attention(h, w, variant, method, a, meter, causal):
    import crypten
    B, N, D = h.size(0), h.size(1), a.hidden
    H, dh = a.heads, a.hidden // a.heads

    def linear(t, W, b):
        return t.matmul(W.t()) + b

    def l2norm(t):
        return t * (t.square().sum(dim=-1, keepdim=True) + 1e-12).inv_sqrt()

    q = meter.run("linear", linear, h, w["q"], w["qb"]).reshape(B, N, H, dh).permute(0, 2, 1, 3)
    k = meter.run("linear", linear, h, w["k"], w["kb"]).reshape(B, N, H, dh).permute(0, 2, 1, 3)
    v = meter.run("linear", linear, h, w["v"], w["vb"]).reshape(B, N, H, dh).permute(0, 2, 1, 3)
    if variant == "qknorm":
        qk = meter.run("qknorm", l2norm, crypten.cat([q, k], dim=0))
        q, k = qk[:B], qk[B:]
        logits = meter.run("linear", lambda q, k: q.matmul(k.transpose(-1, -2)) * w["g"] - w["g"], q, k)
    else:
        logits = meter.run("linear", lambda q, k: q.matmul(k.transpose(-1, -2)).div(math.sqrt(dh)), q, k)
    if causal:
        keep = torch.tril(torch.ones(N, N))
        logits = meter.run("other", lambda t: t * keep + MASK_VALUE * (1 - keep), logits)
    attn = meter.run("softmax", softmax, logits, method)
    return meter.run("attn_mix", lambda a_, v_: linear(a_.matmul(v_).permute(0, 2, 1, 3).reshape(B, N, D),
                                                       w["o"], w["ob"]), attn, v)


def layernorm(t, w, b):
    mean = t.mean(dim=-1, keepdim=True)
    return (t - mean) * (t.var(dim=-1, keepdim=True) + 1e-5).inv_sqrt() * w + b


def run_model(x, W, variant, method, a, meter):
    """x: encrypted hidden states [1, N, D] (bert, gpt2) or encrypted one-hot ids [1, N, V] (bert_full)."""
    def linear(t, Wt, b):
        return t.matmul(Wt.t()) + b

    if a.arch == "bert_full":
        N = x.size(1)
        h = meter.run("embedding", lambda t: t.matmul(W["emb"]) + W["pos"][:N] + W["type"][0], x)
        h = meter.run("layernorm", layernorm, h, W["eln"], W["elnb"])
    else:
        h = x

    for L in range(a.layers):
        w = W["layers"][L]
        if a.arch == "gpt2":   # pre-LN decoder block
            y = meter.run("layernorm", layernorm, h, w["ln1"], w["ln1b"])
            h = meter.run("other", lambda s, t: s + t, h, attention(y, w, variant, method, a, meter, causal=True))
            y = meter.run("layernorm", layernorm, h, w["ln2"], w["ln2b"])
            y = meter.run("linear", linear, y, w["f1"], w["f1b"])
            y = meter.run("gelu", lambda t: t.gelu(), y)
            y = meter.run("linear", linear, y, w["f2"], w["f2b"])
            h = meter.run("other", lambda s, t: s + t, h, y)
        else:                  # BERT post-LN encoder block
            o = attention(h, w, variant, method, a, meter, causal=False)
            h = meter.run("layernorm", layernorm, meter.run("other", lambda s, t: s + t, h, o), w["ln1"], w["ln1b"])
            f = meter.run("linear", linear, h, w["f1"], w["f1b"])
            f = meter.run("gelu", lambda t: t.gelu(), f)
            f = meter.run("linear", linear, f, w["f2"], w["f2b"])
            h = meter.run("layernorm", layernorm, meter.run("other", lambda s, t: s + t, h, f), w["ln2"], w["ln2b"])

    if a.arch == "gpt2":
        h = meter.run("layernorm", layernorm, h, W["lnf"], W["lnfb"])
    if a.arch == "bert_full":
        pooled = meter.run("head", lambda t: linear(t[:, 0], W["pool"], W["poolb"]).tanh(), h)
        h = meter.run("head", lambda t: linear(t, W["cls"], W["clsb"]), pooled)
    return h


def random_weights(a, variant):
    import crypten
    torch.manual_seed(0)
    D, F_ = a.hidden, 4 * a.hidden
    s = 1.0 / math.sqrt(D)
    enc = lambda t: crypten.cryptensor(t, src=0)
    layers = []
    for _ in range(a.layers):
        w = {n: torch.randn(D, D) * s for n in ["q", "k", "v", "o"]}
        w.update({n + "b": torch.zeros(D) for n in ["q", "k", "v", "o"]})
        w.update({"f1": torch.randn(F_, D) * s, "f1b": torch.zeros(F_), "f2": torch.randn(D, F_) * s,
                  "f2b": torch.zeros(D), "ln1": torch.ones(D), "ln1b": torch.zeros(D),
                  "ln2": torch.ones(D), "ln2b": torch.zeros(D)})
        if variant == "qknorm":
            w["g"] = torch.tensor(a.g)
        layers.append({n: enc(t) for n, t in w.items()})
    W = {"layers": layers}
    if a.arch == "gpt2":
        W.update({"lnf": enc(torch.ones(D)), "lnfb": enc(torch.zeros(D))})
    if a.arch == "bert_full":
        W.update({"emb": enc(torch.randn(a.vocab, D) * 0.02), "pos": enc(torch.randn(512, D) * 0.02),
                  "type": enc(torch.randn(2, D) * 0.02), "eln": enc(torch.ones(D)), "elnb": enc(torch.zeros(D)),
                  "pool": enc(torch.randn(D, D) * s), "poolb": enc(torch.zeros(D)),
                  "cls": enc(torch.randn(2, D) * s), "clsb": enc(torch.zeros(2))})
    return W


def model_input(a, N):
    import crypten
    if a.arch == "bert_full":
        ids = torch.randint(0, a.vocab, (1, N))
        return crypten.cryptensor(torch.nn.functional.one_hot(ids, a.vocab).float(), src=1)
    return crypten.cryptensor(torch.randn(1, N, a.hidden), src=1)


def main():
    import crypten
    a = ARGS
    crypten.init()
    rank = crypten.comm.get().get_rank()
    host = socket.gethostname()
    gbps = rtt_ms = float("nan")
    if a.net_probe:
        gbps, rtt_ms = net_probe(rank)
        print(f"[rank {rank} {host}] link: {gbps:.2f} Gbps, round trip {rtt_ms:.3f} ms", flush=True)
    rows = []
    for N in [int(v) for v in a.seq_lens.split(",")]:
        for spec in a.configs.split(","):
            variant, method = spec.split(":")
            W = random_weights(a, variant)
            x = model_input(a, N)
            meter = Meter()
            crypten.reset_communication_stats()
            t0 = time.time()
            run_model(x, W, variant, method, a, meter)
            dt = time.time() - t0
            st = crypten.get_communication_stats()
            row = {"arch": a.arch, "variant": variant, "softmax_method": method, "layers": a.layers,
                   "hidden": a.hidden, "heads": a.heads, "seq_len": N, "host": host,
                   "multinode": os.environ.get("SNM_MULTINODE") == "1", "link_gbps": gbps, "link_rtt_ms": rtt_ms,
                   "total_rounds": st["rounds"], "total_bytes": st["bytes"], "total_time_s": dt,
                   "comm_time_s": float(st.get("time", 0.0))}
            for c in CATS:
                r, b, t = meter.acc.get(c, [0, 0, 0.0])
                row[f"{c}_rounds"], row[f"{c}_bytes"], row[f"{c}_time_s"] = r, b, t
            rows.append(row)
            if rank == 0:
                print(f"[{a.arch} N={N:4d} {variant:7s} {method:11s}] rounds {st['rounds']:6d} "
                      f"bytes {st['bytes']/1e9:8.3f} GB time {dt:8.1f}s | softmax {row['softmax_rounds']} rd "
                      f"{row['softmax_bytes']/1e9:.3f} GB | qknorm {row['qknorm_rounds']} rd | "
                      f"embedding {row['embedding_bytes']/1e9:.3f} GB | head {row['head_bytes']/1e9:.3f} GB", flush=True)
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[done] {a.out}")


if __name__ == "__main__":
    if os.environ.get("SNM_MULTINODE") == "1":
        # one party per machine: RANK, WORLD_SIZE and RENDEZVOUS come from the launcher (e.g. srun)
        main()
    else:
        from multiprocess_launcher import MultiProcessLauncher
        launcher = MultiProcessLauncher(2, main)
        launcher.start()
        launcher.join()
        launcher.terminate()

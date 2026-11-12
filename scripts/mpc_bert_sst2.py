#!/usr/bin/env python3
"""End-to-end 2-party secure inference of a fine-tuned BERT-base SST-2 classifier.

First secure run of a *converted* BERT: trained weights (not random), real SST-2 validation sentences, secure
accuracy against the plaintext model on the same examples. Weights held by party 0 and secret-shared; the
embedding output of each sentence is computed by the client (party 1) in the clear and then encrypted by
party 1, which reveals the sentence's embedding to nobody else; encoder, pooler and classifier are fully
secure. Sentences run at their true length with no padding, so every row has all N keys visible.

Softmax variants (method):
  crypten_max   vanilla BERT, secure max + exp + reciprocal (CrypTen)
  shaft_ode     vanilla BERT, clip + ODE (SHAFT)
  snm_shifted   QK-norm BERT, logits g*cos - g in [-2g, 0], exp + reciprocal, no comparison; public rescale
                c = max(1, N/64) of the row sum (0 rounds)

Per example one CSV row is appended and flushed, so a killed job loses nothing; --start/--end select a range.

Usage (from SHAFT's examples/unit-test directory):
    python mpc_bert_sst2.py --export export/bert_qknorm_g30_sst2.pt --method snm_shifted \
        --out results/mpc/mpc_bert_sst2_qknorm_g30.csv
"""

import argparse
import csv
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("SHAFT_UNIT_TEST_DIR", os.getcwd()))

_ap = argparse.ArgumentParser()
_ap.add_argument("--export", required=True)
_ap.add_argument("--method", required=True, choices=["crypten_max", "shaft_ode", "snm_shifted"])
_ap.add_argument("--start", type=int, default=0)
_ap.add_argument("--end", type=int, default=10**9)
_ap.add_argument("--precision_bits", type=int, default=16)
_ap.add_argument("--debug", action="store_true", help="decrypt and print activation ranges for the first example (leaks, debugging only)")
_ap.add_argument("--out", required=True)
ARGS = _ap.parse_known_args()[0]

HEADS, HIDDEN = 12, 768


def softmax(logits, method, n):
    from crypten.config import cfg
    if method == "crypten_max":
        with cfg.temp_override({"functions.softmax_method": "reciprocal"}):
            return logits.softmax(-1)
    if method == "shaft_ode":
        with cfg.temp_override({"functions.softmax_method": "ode", "functions.softmax_ode_clip": True,
                                "functions.softmax_ode_lb": -4.0, "functions.softmax_ode_ub": 12.0,
                                "functions.softmax_ode_iter_num": 16}):
            return logits.softmax(-1)
    over = {"functions.softmax_method": "shifted"}
    c = max(1.0, n / 64.0)
    if c != 1.0:
        over["functions.softmax_shift_rescale"] = c
    with cfg.temp_override(over):
        return logits.softmax(-1)


def layernorm(t, w, b, c2):
    # inv_sqrt(var) = inv_sqrt(var / c2) / c: the public scale c2 keeps CrypTen's Newton iteration inside its range
    mean = t.mean(dim=-1, keepdim=True)
    return (t - mean) * (t.var(dim=-1, keepdim=True).div(c2) + 1e-5 / c2).inv_sqrt() * (w * (1.0 / math.sqrt(c2))) + b


def forward(h, W, qk, method, dbg=False):
    import crypten
    rank = crypten.comm.get().get_rank()

    def show(tag, t):
        if dbg:
            p = t.get_plain_text()
            if rank == 0:
                print(f'[dbg] {tag}: min {float(p.min()):.4g} max {float(p.max()):.4g} finite {bool(torch.isfinite(p).all())}', flush=True)
    B, N, D = h.size(0), h.size(1), HIDDEN
    dh = D // HEADS

    def linear(t, Wt, b):
        return t.matmul(Wt.t()) + b

    def heads(t):
        return t.reshape(B, N, HEADS, dh).permute(0, 2, 1, 3)

    def l2norm(t, c2):
        x = t * (1.0 / math.sqrt(c2))     # public scale: x / ||x|| is invariant, and ||x||^2 stays in Newton's range
        return x * (x.square().sum(dim=-1, keepdim=True) + 1e-12).inv_sqrt()

    for li, w in enumerate(W["layers"]):
        q, k, v = heads(linear(h, w["q"], w["qb"])), heads(linear(h, w["k"], w["kb"])), heads(linear(h, w["v"], w["vb"]))
        if qk:
            show(f'L{li} q sumsq', q.square().sum(dim=-1)); show(f'L{li} k sumsq', k.square().sum(dim=-1))
            qkn = l2norm(crypten.cat([q, k], dim=0), float(w['qk_c2']))
            q, k = qkn[:B], qkn[B:]
            show(f'L{li} qn norm', qkn.square().sum(dim=-1))
            logits = q.matmul(k.transpose(-1, -2)) * w["g"] - w["g"]
        else:
            logits = q.matmul(k.transpose(-1, -2)).div(math.sqrt(dh))
        show(f'L{li} logits', logits)
        attn = softmax(logits, method, N)
        show(f'L{li} attn rowsum', attn.sum(dim=-1))
        o = linear(attn.matmul(v).permute(0, 2, 1, 3).reshape(B, N, D), w["o"], w["ob"])
        show(f'L{li} ln1 var', (h + o).var(dim=-1))
        h = layernorm(h + o, w["ln1"], w["ln1b"], float(w["ln1_c2"]))
        show(f'L{li} h after ln1', h)
        pre = linear(h, w["f1"], w["f1b"])
        show(f'L{li} ffn pre-gelu', pre)
        f = linear(pre.gelu(), w["f2"], w["f2b"])
        show(f'L{li} ln2 var', (h + f).var(dim=-1))
        h = layernorm(h + f, w["ln2"], w["ln2b"], float(w["ln2_c2"]))
        show(f'L{li} h out', h)
    pooled = linear(h[:, 0], W["pool"], W["poolb"]).tanh()
    return linear(pooled, W["cls"], W["clsb"])


def main():
    import crypten
    from crypten.config import cfg

    a = ARGS
    crypten.init()
    rank = crypten.comm.get().get_rank()
    ex = torch.load(a.export, map_location="cpu")
    qk = ex["variant"] == "qknorm"
    if (a.method == "snm_shifted") != qk:
        raise SystemExit(f"method {a.method} does not match export variant {ex['variant']}")

    fields = ["idx", "n_tokens", "label", "ref_logit0", "ref_logit1", "sec_logit0", "sec_logit1", "ref_pred",
              "sec_pred", "max_logit_err", "finite", "rounds", "bytes", "time_s", "method", "precision_bits"]
    new = rank == 0 and not os.path.exists(a.out)
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    f = open(a.out, "a", newline="") if rank == 0 else None
    wr = csv.DictWriter(f, fieldnames=fields) if rank == 0 else None
    if new:
        wr.writeheader()

    with cfg.temp_override({"encoder.precision_bits": a.precision_bits}):
        enc = lambda t: crypten.cryptensor(t, src=0)
        PUB = ("qk_c2", "ln1_c2", "ln2_c2")
        W = {"layers": [{n: (t if n in PUB else enc(t)) for n, t in L.items()} for L in ex["weights"]["layers"]]}
        W.update({n: enc(ex["weights"][n]) for n in ["pool", "poolb", "cls", "clsb"]})
        for i, e in enumerate(ex["examples"]):
            if i < a.start or i >= a.end:
                continue
            crypten.reset_communication_stats()
            t0 = time.time()
            x = crypten.cryptensor(e["h0"].unsqueeze(0), src=1)
            out = forward(x, W, qk, a.method, dbg=a.debug and i == a.start).get_plain_text()[0]
            dt = time.time() - t0
            st = crypten.get_communication_stats()
            ref = e["ref_logits"]
            ok = bool(torch.isfinite(out).all())
            if rank == 0:
                wr.writerow({"idx": e["idx"], "n_tokens": e["ids"].numel(), "label": e["label"],
                             "ref_logit0": float(ref[0]), "ref_logit1": float(ref[1]),
                             "sec_logit0": float(out[0]), "sec_logit1": float(out[1]),
                             "ref_pred": int(ref.argmax()), "sec_pred": int(out.argmax()),
                             "max_logit_err": float((out - ref).abs().max()) if ok else float("nan"),
                             "finite": ok, "rounds": st["rounds"], "bytes": st["bytes"], "time_s": dt,
                             "method": a.method, "precision_bits": a.precision_bits})
                f.flush()
                print(f"[{a.method}] ex {i} N={e['ids'].numel()} ref {int(ref.argmax())} sec {int(out.argmax())} "
                      f"err {float((out - ref).abs().max()):.3f} {dt:.1f}s", flush=True)
    if f:
        f.close()


if __name__ == "__main__":
    from multiprocess_launcher import MultiProcessLauncher
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()

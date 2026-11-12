#!/usr/bin/env python3
"""Secure (2-party MPC) softmax with and without comparisons, on provably bounded inputs.

In MPC every comparison is a protocol with its own communication rounds, and the numerically
stable softmax needs n-1 of them per row to find the maximum. When the architecture bounds the
logits (scalar-gain QK-norm: |logit| <= g; soft-capping: |logit| < c), the shift can be a public
constant and the softmax needs no comparison at all. Obliviousness matters here: a data-dependent
guard would have to run the fallback on every row, so in MPC only a *provable* bound helps.

Variants, all on SHAFT's CrypTen fork (NDSS 2025), same inputs:
  crypten_max    CrypTen's softmax: secure max (log-reduction tree), exp, Newton reciprocal
  snm_public     Soft-no-Max: subtract the public bound m* = g, exp, Newton reciprocal (0 comparisons)
  shaft_ode      SHAFT: clip every element to [-4, 12] (2 ReLUs each), ODE with t = 16
  ode_bounded    SHAFT's ODE without clipping, t = ceil(2g): Theorem 1's precondition
                 max - min <= t holds by the bound (0 comparisons)

Rows are drawn inside [-g, g] in two regimes: "cosine" (g * cos of random 64-d directions, diffuse
like most attention rows) and "peaked" (one entry near +g, the rest spread below). Rounds and bytes
are exact and hardware-independent; time is indicative on a shared machine.

Usage (from SHAFT's examples/unit-test directory, for its MultiProcessLauncher):
    python mpc_softmax_bench.py --g 4.75 --lengths 32,64,128,256,512 --rows 16 --out mpc_softmax.csv
"""

import argparse
import csv
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("SHAFT_UNIT_TEST_DIR", os.getcwd()))


def make_rows(regime, rows, n, g, seed):
    gen = torch.Generator().manual_seed(seed)
    if regime == "cosine":
        q = torch.nn.functional.normalize(torch.randn(rows, 1, 64, generator=gen), dim=-1)
        k = torch.nn.functional.normalize(torch.randn(rows, n, 64, generator=gen), dim=-1)
        x = g * (q * k).sum(-1)
    elif regime == "peaked":
        x = -g + (g * torch.rand(rows, n, generator=gen))          # in [-g, 0)
        top = torch.randint(0, n, (rows,), generator=gen)
        x[torch.arange(rows), top] = g * (0.8 + 0.2 * torch.rand(rows, generator=gen))
    elif regime == "flat_low":
        # Worst case for a public shift: the row maximum sits far below the bound, so every
        # exp(x - g) is small and the denominator the reciprocal must invert is tiny.
        x = -g + 0.5 * g * torch.rand(rows, n, generator=gen)      # in [-g, -g/2)
    else:
        raise ValueError(regime)
    return x.clamp(-g, g)


def snm_public(enc, m_star, dim=-1):
    import crypten
    from crypten.config import cfg
    numerator = (enc - m_star).exp()
    with cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inv = numerator.sum(dim, keepdim=True).reciprocal()
    return numerator * inv


def main():
    import crypten
    from crypten.config import cfg

    a = ARGS
    crypten.init()
    rank = crypten.comm.get().get_rank()
    t_bounded = max(1, math.ceil(2 * a.g))
    variants = {
        "crypten_max": ({"functions.softmax_method": "reciprocal"}, None),
        "snm_public": ({}, "snm"),
        "snm_public_nr20": ({"functions.reciprocal_nr_iters": 20}, "snm"),
        "shaft_ode": ({"functions.softmax_method": "ode", "functions.softmax_ode_clip": True,
                       "functions.softmax_ode_lb": -4.0, "functions.softmax_ode_ub": 12.0,
                       "functions.softmax_ode_iter_num": 16}, None),
        "ode_bounded": ({"functions.softmax_method": "ode", "functions.softmax_ode_clip": False,
                         "functions.softmax_ode_iter_num": t_bounded}, None),
    }
    results = []
    for prec in [int(p) for p in a.precision_bits.split(",")]:
        for regime in a.regimes.split(","):
            for n in [int(v) for v in a.lengths.split(",")]:
                x = make_rows(regime, a.rows, n, a.g, seed=1000 + n)
                ref = torch.softmax(x.double(), -1)
                for name, (override, special) in variants.items():
                    with cfg.temp_override({"encoder.precision_bits": prec, **override}):
                        enc = crypten.cryptensor(x, src=0)
                        crypten.reset_communication_stats()
                        t0 = time.time()
                        out = snm_public(enc, a.g) if special == "snm" else enc.softmax(-1)
                        dt = time.time() - t0
                        st = crypten.get_communication_stats()
                        y = out.get_plain_text().double()
                    err = (y - ref).abs()
                    kl = (ref * (ref.clamp_min(1e-300).log() - y.clamp_min(1e-12).log())).sum(-1)
                    results.append({
                        "variant": name, "precision_bits": prec, "regime": regime, "n": n,
                        "rows": a.rows, "g": a.g, "ode_t": override.get("functions.softmax_ode_iter_num", ""),
                        "rounds": st["rounds"], "bytes": st["bytes"], "time_s": dt,
                        "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
                        "argmax_agree": float((y.argmax(-1) == ref.argmax(-1)).double().mean()),
                        "mean_kl": float(kl.mean()), "row_sum_dev": float((y.sum(-1) - 1).abs().max()),
                    })
                    if rank == 0:
                        r = results[-1]
                        print(f"prec={prec:2d} {regime:6s} n={n:4d} {name:12s} rounds={r['rounds']:5d} "
                              f"bytes={r['bytes']/1e6:9.3f}MB time={dt:7.3f}s max|err|={r['max_abs_err']:.2e} "
                              f"argmax={r['argmax_agree']:.3f} KL={r['mean_kl']:.2e}", flush=True)
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"[done] {a.out}")


# Parsed at import: the launcher spawns each party as a fresh interpreter that re-imports this
# module (with the parent's argv) but does not run the __main__ block.
_ap = argparse.ArgumentParser()
_ap.add_argument("--g", type=float, default=4.75)
_ap.add_argument("--lengths", default="32,64,128,256,512")
_ap.add_argument("--rows", type=int, default=16)
_ap.add_argument("--regimes", default="cosine,peaked")
_ap.add_argument("--precision_bits", default="16")
_ap.add_argument("--out", default="mpc_softmax.csv")
ARGS = _ap.parse_known_args()[0]

if __name__ == "__main__":
    from multiprocess_launcher import MultiProcessLauncher
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()

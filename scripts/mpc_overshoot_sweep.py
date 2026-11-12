#!/usr/bin/env python3
"""How badly does an empirical constant shift fail in secure softmax when the data exceeds it?

NEXUS-style constant shifts are calibrated on evaluation data; out of distribution, a row's maximum
can exceed the constant. In plaintext FP32 a small excess is harmless (Soft-no-Max is exact while
exponentials stay finite). In fixed-point secure computation it is not: the exponential is
approximated as (1 + x/2^t)^(2^t), accurate only for x <= 0 near zero, and the Newton reciprocal of
the denominator starts from an initial guess that assumes a bounded range. This sweeps the excess
delta = row maximum - shift and measures the secure error against the plaintext softmax.

Measured reference for delta: profiled m* was exceeded by 9.2-9.4 nats out of domain on code for
SmolLM2 and TinyLlama (EVIDENCE.md 9c). A provable bound means delta <= 0 for every input.

Variants (2-party, SHAFT's CrypTen fork):
  shift_const  subtract a public constant c = row max - delta (no maximum, no guard)
  crypten_max  secure maximum (reference)

Usage (from SHAFT's examples/unit-test directory):
    python mpc_overshoot_sweep.py --n 128 --rows 32 --deltas -2,0,0.5,1,2,4,6,8,9.3 \
        --out results/mpc/mpc_overshoot_sweep.csv
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.environ.get("SHAFT_UNIT_TEST_DIR", os.getcwd()))

_ap = argparse.ArgumentParser()
_ap.add_argument("--n", type=int, default=128)
_ap.add_argument("--rows", type=int, default=32)
_ap.add_argument("--deltas", default="-2,0,0.5,1,2,4,6,8,9.3")
_ap.add_argument("--precision_bits", default="16,20")
_ap.add_argument("--out", default="results/mpc/mpc_overshoot_sweep.csv")
ARGS = _ap.parse_known_args()[0]


def rows_with_max(rows, n, gen):
    """Attention-like rows: a few strong entries and a diffuse tail, row maximum exactly 0."""
    x = -6.0 + 3.0 * torch.rand(rows, n, generator=gen)
    k = max(1, n // 32)
    top = torch.rand(rows, n, generator=gen).argsort(-1)[:, :k]
    x.scatter_(1, top, -1.5 * torch.rand(rows, k, generator=gen))
    return x - x.max(-1, keepdim=True).values


def main():
    import crypten
    from crypten.config import cfg

    a = ARGS
    crypten.init()
    rank = crypten.comm.get().get_rank()
    gen = torch.Generator().manual_seed(0)
    base = rows_with_max(a.rows, a.n, gen)           # row maximum 0
    ref = torch.softmax(base.double(), -1)            # softmax is shift-invariant: same for every delta
    results = []
    for prec in [int(p) for p in a.precision_bits.split(",")]:
        for delta in [float(d) for d in a.deltas.split(",")]:
            # the data exceed the constant shift by delta: logits = base + delta, shift = 0
            x = base + delta
            for variant in ["shift_const", "crypten_max"]:
                with cfg.temp_override({"encoder.precision_bits": prec}):
                    enc = crypten.cryptensor(x, src=0)
                    crypten.reset_communication_stats()
                    if variant == "crypten_max":
                        with cfg.temp_override({"functions.softmax_method": "reciprocal"}):
                            out = enc.softmax(-1)
                    else:
                        num = enc.exp()                                   # shift c = 0, overshoot delta
                        with cfg.temp_override({"functions.reciprocal_all_pos": True}):
                            out = num * num.sum(-1, keepdim=True).reciprocal()
                    st = crypten.get_communication_stats()
                    y = out.get_plain_text().double()
                finite = torch.isfinite(y).all(-1)
                err = (y - ref).abs()
                row = {"precision_bits": prec, "delta_nats": delta, "variant": variant, "n": a.n, "rows": a.rows,
                       "max_abs_err": float(err[finite].max()) if finite.any() else float("inf"),
                       "mean_abs_err": float(err[finite].mean()) if finite.any() else float("inf"),
                       "argmax_agree": float((y.argmax(-1) == ref.argmax(-1)).double().mean()),
                       "max_row_sum_dev": float((y.sum(-1) - 1).abs().max()),
                       "nonfinite_rows": int((~finite).sum()), "rounds": st["rounds"], "bytes": st["bytes"]}
                results.append(row)
                if rank == 0:
                    print(f"prec={prec} delta={delta:+5.1f} {variant:11s} max|err| {row['max_abs_err']:.2e} "
                          f"argmax {row['argmax_agree']:.3f} sum dev {row['max_row_sum_dev']:.2e} "
                          f"nonfinite {row['nonfinite_rows']}", flush=True)
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"[done] {a.out}")


if __name__ == "__main__":
    from multiprocess_launcher import MultiProcessLauncher
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()

#!/usr/bin/env python3
"""Empirical constant shift (NEXUS-style) vs provable bound, in secure 2-party softmax.

NEXUS (NDSS 2025) replaces the row maximum with a constant chosen because "all inputs are in this
range" on its evaluation data. A provable bound (scalar-gain QK-norm: |logit| <= g) holds for
every input. This experiment shows where the difference matters:

1. Plaintext: run the trained QK-norm CIFAR-10 ViT on clean test images and record every attention
   logit. Calibrate the empirical constant c_emp = max clean logit (+ --margin), as an evaluation-
   data range would be chosen.
2. Run the same model on out-of-distribution inputs (Gaussian noise, constant colour, contrast x4,
   inverted, and a clean control) and count rows whose maximum exceeds c_emp, with the overshoot.
   The provable bound g covers all of them by construction.
3. Secure: on a sample of attention rows from each input set, run 2-party softmax with
     emp_const   shift by c_emp (no maximum, no bound): exponential arguments can be positive
     provable_g  shift by g (no maximum, provable)
     crypten_max secure maximum (reference)
   and compare against the plaintext softmax (max |error|, argmax agreement, row-sum deviation).

Usage (from SHAFT's examples/unit-test directory):
    python mpc_empirical_vs_provable.py --ckpt results/ckpt100/vit_cifar_qknorm.pt \
        --data data/cifar10.npz --n_images 256 --rows 64 --out results/mpc/mpc_empirical_vs_provable.csv
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("SHAFT_UNIT_TEST_DIR", os.getcwd()))

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default="results/ckpt100/vit_cifar_qknorm.pt")
_ap.add_argument("--data", default="data/cifar10.npz")
_ap.add_argument("--n_images", type=int, default=256)
_ap.add_argument("--rows", type=int, default=64, help="attention rows per input set for the secure part")
_ap.add_argument("--margin", type=float, default=0.0, help="added to the clean maximum for c_emp")
_ap.add_argument("--precision_bits", type=int, default=16)
_ap.add_argument("--out", default="results/mpc/mpc_empirical_vs_provable.csv")
ARGS = _ap.parse_known_args()[0]


def input_sets(x_clean, gen):
    """Clean control plus out-of-distribution variants, all in the model's normalised input space."""
    n = x_clean.shape[0]
    return {
        "clean": x_clean,
        "gaussian_noise": torch.randn(x_clean.shape, generator=gen) * 3.0,
        "constant_colour": torch.randn(n, 3, 1, 1, generator=gen).expand_as(x_clean).contiguous() * 2.0,
        "contrast_x4": x_clean * 4.0,
        "contrast_x8": x_clean * 8.0,
        "contrast_x16": x_clean * 16.0,
        "inverted": -x_clean,
    }


def attention_rows(model, x, qknorm):
    """Every attention logit row of every block, as [rows, n_tokens]; the provable g bound if any."""
    rows = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 128):
            model(x[i:i + 128])
            for blk in model.blocks:
                lg = blk.attn.last_logits            # B, heads, N, N
                rows.append(lg.reshape(-1, lg.shape[-1]))
    g = max(float(b.attn.g().max()) for b in model.blocks) if qknorm else float("nan")
    return torch.cat(rows), g


def main():
    import crypten
    from crypten.config import cfg
    from vit_qknorm_cifar import ViT, _normalise

    a = ARGS
    crypten.init()
    rank = crypten.comm.get().get_rank()

    ck = torch.load(a.ckpt, map_location="cpu")
    qk = bool(ck["qknorm"])   # vanilla: no provable bound, only the empirical constant and the secure max
    model = ViT(qk)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    d = np.load(a.data)
    _, _, xte, _ = _normalise(d["xtr"], d["ytr"], d["xte"], d["yte"])
    x_clean = xte[: a.n_images].float()

    gen = torch.Generator().manual_seed(0)
    sets = input_sets(x_clean, gen)
    clean_rows, g = attention_rows(model, sets["clean"], qk)
    c_emp = float(clean_rows.max()) + a.margin
    variants = ["emp_const", "provable_g", "crypten_max"] if qk else ["emp_const", "crypten_max"]

    results = []
    for name, x in sets.items():
        rows_all, _ = attention_rows(model, x, qk)
        row_max = rows_all.max(-1).values
        over = row_max > c_emp
        # the secure part uses the rows most exposed to the empirical shift (largest maxima)
        idx = torch.argsort(row_max, descending=True)[: a.rows]
        sample = rows_all[idx].float()
        ref = torch.softmax(sample.double(), -1)
        stats = {"input_set": name, "g_bound": g, "c_emp": c_emp, "n_rows": int(rows_all.shape[0]),
                 "rows_over_c_emp": int(over.sum()), "frac_over_c_emp": float(over.float().mean()),
                 "max_row_max": float(row_max.max()), "max_overshoot": float((row_max - c_emp).clamp_min(0).max()),
                 "bound_holds": bool(float(rows_all.abs().max()) <= g + 1e-4) if qk else None}

        for variant in variants:
            with cfg.temp_override({"encoder.precision_bits": a.precision_bits}):
                enc = crypten.cryptensor(sample, src=0)
                if variant == "crypten_max":
                    with cfg.temp_override({"functions.softmax_method": "reciprocal"}):
                        out = enc.softmax(-1)
                else:
                    shift = c_emp if variant == "emp_const" else g
                    num = (enc - shift).exp()
                    with cfg.temp_override({"functions.reciprocal_all_pos": True}):
                        out = num * num.sum(-1, keepdim=True).reciprocal()
                y = out.get_plain_text().double()
            err = (y - ref).abs()
            finite = torch.isfinite(y).all(-1)
            row = dict(stats)
            row.update({"variant": variant, "max_abs_err": float(err[finite].max()) if finite.any() else float("inf"),
                        "mean_abs_err": float(err[finite].mean()) if finite.any() else float("inf"),
                        "nonfinite_rows": int((~finite).sum()),
                        "argmax_agree": float((y.argmax(-1) == ref.argmax(-1)).double().mean()),
                        "max_row_sum_dev": float((y.sum(-1) - 1).abs().max())})
            results.append(row)
            if rank == 0:
                print(f"[{name:15s} {variant:11s}] rows>{c_emp:.3f}: {stats['rows_over_c_emp']}/{stats['n_rows']} "
                      f"(max overshoot {stats['max_overshoot']:.3f}, bound g={g:.3f} holds {stats['bound_holds']}) | "
                      f"max|err| {row['max_abs_err']:.2e} argmax {row['argmax_agree']:.3f} "
                      f"sum dev {row['max_row_sum_dev']:.2e} nonfinite {row['nonfinite_rows']}", flush=True)

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

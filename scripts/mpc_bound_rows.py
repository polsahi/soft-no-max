#!/usr/bin/env python3
"""Is the bounded, comparison-free secure softmax safe for EVERY row once the row sum is rescaled?

Under scalar-gain QK-norm every shifted logit lies in [-2g, 0], so a row of N visible keys sums to
s in [e^-2g, N] (e^-2g: a causal first row with one key at the bottom of the range). CrypTen's
Newton reciprocal is only accurate inside a window of ln s (MPC_REFRAME.md section 4g', caveat), so
the claim "provably stable" needs the whole range [-2g, ln N] to fit that window after a public
rescale c: 1/s = (1/(s/c))/c.

2-party CrypTen (SHAFT's fork, patched by shaft_apply_patches.py), run from SHAFT's
examples/unit-test directory:

  part 1  window   secure 1/s against the truth on a fine grid of ln s; the contiguous range
                   around 0 where every repetition is within --tol is the window [lo, hi]
  part 2  rows     worst-case and typical rows for every g in --gs and N in --ns, secure softmax
                   ("shifted") with the rescale off (c = 1) and on (c centres [-2g, ln N] in the
                   window); max |p - softmax|, max row-sum error, and whether the output is garbage

Usage:
    python mpc_bound_rows.py --gs 3.0,5.5 --ns 32,64,128,256,512 --out_dir results/mpc
"""

import argparse
import csv
import math
import os
import sys

import torch

sys.path.insert(0, os.environ.get("SHAFT_UNIT_TEST_DIR", os.getcwd()))

_ap = argparse.ArgumentParser()
_ap.add_argument("--gs", default="3.0,5.5")
_ap.add_argument("--ns", default="32,64,128,256,512")
_ap.add_argument("--precision_bits", type=int, default=16)
_ap.add_argument("--nr_iters", type=int, default=0, help="Newton reciprocal iterations (0: CrypTen default, 10)")
_ap.add_argument("--tol", type=float, default=0.01)
_ap.add_argument("--reps", type=int, default=64)
_ap.add_argument("--grid", default="-11,6.5,0.25", help="ln s from,to,step for the window sweep")
_ap.add_argument("--rows", type=int, default=16, help="random rows per (type, g, N)")
_ap.add_argument("--configs", default="shifted:1,shifted:top,crypten_max",
                 help="softmax:rescale, rescale in {1, top = max(1, N/64), center}; crypten_max = secure max")
_ap.add_argument("--out_dir", default="results/mpc")
ARGS = _ap.parse_known_args()[0]

MASKED = -50.0          # public mask value for keys a causal row cannot see


def window(crypten, cfg, rank):
    lo_t, hi_t, step = (float(v) for v in ARGS.grid.split(","))
    ts = torch.arange(lo_t, hi_t + 1e-9, step, dtype=torch.float64)
    s = ts.exp().float()
    with cfg.temp_override({"functions.reciprocal_all_pos": True}):
        rec = crypten.cryptensor(s.repeat(ARGS.reps, 1), src=0).reciprocal().get_plain_text()
    rel = (rec * s - 1).abs()                                  # reps, grid
    ok = (rel <= ARGS.tol).all(0)
    zero = int((ts.abs()).argmin())
    lo_i = hi_i = zero
    while lo_i > 0 and ok[lo_i - 1]:
        lo_i -= 1
    while hi_i < len(ts) - 1 and ok[hi_i + 1]:
        hi_i += 1
    rows = [{"ln_s": float(t), "s": float(math.exp(t)), "median_rel_err": float(rel[:, j].median()),
             "max_rel_err": float(rel[:, j].max()), "all_within_tol": bool(ok[j])} for j, t in enumerate(ts)]
    lo, hi = float(ts[lo_i]), float(ts[hi_i])
    if rank == 0:
        print(f"[window] {ARGS.precision_bits}-bit, default Newton iterations: every one of {ARGS.reps} "
              f"repetitions within {ARGS.tol:.0%} for ln s in [{lo:+.2f}, {hi:+.2f}] "
              f"(width {hi - lo:.2f} nats)", flush=True)
        for r in rows:
            print(f"  ln s {r['ln_s']:+6.2f}  median rel err {r['median_rel_err']:.2e}  max {r['max_rel_err']:.2e}"
                  f"  {'ok' if r['all_within_tol'] else '--'}", flush=True)
    return lo, hi, rows


def make_rows(kind, g, n, count, gen):
    """Shifted logits in [-2g, 0] (masked entries at MASKED), shape [count, n]."""
    lo = -2.0 * g
    if kind == "all_bottom":          # s = N e^-2g
        x = torch.full((count, n), lo)
    elif kind == "all_top":           # s = N: the upper end
        x = torch.zeros(count, n)
    elif kind == "one_top":           # s = 1 + (N-1) e^-2g: a sharp row
        x = torch.full((count, n), lo); x[:, 0] = 0.0
    elif kind == "causal_first":      # s = e^-2g: one visible key, at the bottom
        x = torch.full((count, n), MASKED); x[:, 0] = lo
    elif kind == "uniform":           # logits uniform over the whole range
        x = lo * torch.rand(count, n, generator=gen)
    elif kind == "trained_like":      # cosines ~ N(0.1, 0.15) clipped, one related key at 0.9
        c = (0.1 + 0.15 * torch.randn(count, n, generator=gen)).clamp(-1, 1)
        c[:, 0] = 0.9
        x = g * c - g
    else:
        raise ValueError(kind)
    return x


def rescale_for(g, n, lo, hi):
    """Public c that centres ln s in [-2g, ln n] inside the window [lo, hi]."""
    return math.exp((math.log(n) - 2 * g) / 2 - (lo + hi) / 2)


def main():
    import crypten
    from crypten.config import cfg

    crypten.init()
    rank = crypten.comm.get().get_rank()
    out = []
    ov = {"encoder.precision_bits": ARGS.precision_bits}
    if ARGS.nr_iters:
        ov["functions.reciprocal_nr_iters"] = ARGS.nr_iters
    with cfg.temp_override(ov):
        lo, hi, wrows = window(crypten, cfg, rank)
        gen = torch.Generator().manual_seed(0)
        kinds = ["all_bottom", "all_top", "one_top", "causal_first", "uniform", "trained_like"]
        for g in (float(v) for v in ARGS.gs.split(",")):
            for n in (int(v) for v in ARGS.ns.split(",")):
                width = 2 * g + math.log(n)
                fits = width <= hi - lo
                for conf in ARGS.configs.split(","):
                    method, _, rule = conf.partition(":")
                    c = {"": 1.0, "1": 1.0, "top": max(1.0, n / 64.0),
                         "center": rescale_for(g, n, lo, hi)}[rule]
                    over = ({"functions.softmax_method": "shifted", "functions.softmax_shift_rescale": c}
                            if method == "shifted" else {"functions.softmax_method": "reciprocal"})
                    for kind in kinds:
                        x = make_rows(kind, g, n, ARGS.rows, gen)
                        ref = torch.softmax(x.double(), -1)
                        crypten.reset_communication_stats()
                        with cfg.temp_override(over):
                            p = crypten.cryptensor(x, src=0).softmax(-1).get_plain_text().double()
                        st = crypten.get_communication_stats()
                        finite = bool(torch.isfinite(p).all())
                        err = float((p - ref).abs().max()) if finite else float("inf")
                        sum_err = float((p.sum(-1) - 1).abs().max()) if finite else float("inf")
                        garbage = (not finite) or err > 0.5 or sum_err > 0.5
                        row = {"g": g, "n": n, "range_width_nats": width, "window_lo": lo, "window_hi": hi,
                               "fits_window": fits, "method": method, "rescale_rule": rule or "1",
                               "rescale": c, "rescaled": c != 1.0, "row_kind": kind,
                               "ln_s_min": math.log(float(torch.exp(x.double()).sum(-1).min())),
                               "ln_s_max": math.log(float(torch.exp(x.double()).sum(-1).max())),
                               "max_abs_err": err, "max_rowsum_err": sum_err, "garbage": garbage,
                               "rounds": st["rounds"], "bytes": st["bytes"]}
                        out.append(row)
                        if rank == 0:
                            print(f"[g={g:.1f} N={n:4d} {method:11s} c={c:6.2f}] {kind:12s} ln s [{row['ln_s_min']:+6.2f}, {row['ln_s_max']:+6.2f}] "
                                  f"max|p-ref| {err:.2e} rowsum {sum_err:.2e} {'GARBAGE' if garbage else ''} "
                                  f"| {st['rounds']} rd", flush=True)
    if rank == 0:
        os.makedirs(ARGS.out_dir, exist_ok=True)
        for name, rows in (("mpc_recip_window.csv", wrows), ("mpc_bound_rows.csv", out)):
            with open(os.path.join(ARGS.out_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"[done] {ARGS.out_dir}/mpc_recip_window.csv, {ARGS.out_dir}/mpc_bound_rows.csv", flush=True)


if __name__ == "__main__":
    from multiprocess_launcher import MultiProcessLauncher
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()

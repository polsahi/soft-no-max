#!/usr/bin/env python3
"""Summarise secure BERT SST-2 runs: secure vs plaintext accuracy and agreement, logit error, cost per sentence.

Usage: python mpc_bert_sst2_summary.py results/mpc/mpc_bert_sst2_*.csv
"""
import csv
import math
import statistics as st
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        continue
    n = len(rows)
    lab = [int(r["label"]) for r in rows]
    ref = [int(r["ref_pred"]) for r in rows]
    sec = [int(r["sec_pred"]) for r in rows]
    fin = sum(r["finite"] == "True" for r in rows)
    errs = [float(r["max_logit_err"]) for r in rows if r["finite"] == "True"]
    acc_p = sum(a == b for a, b in zip(ref, lab)) / n
    acc_s = sum(a == b for a, b in zip(sec, lab)) / n
    agree = sum(a == b for a, b in zip(ref, sec))
    lo, hi = wilson(agree, n)
    print(f"{path.split('/')[-1]}  method={rows[0]['method']} n={n} finite={fin}")
    print(f"  plaintext acc {acc_p:.3f}  secure acc {acc_s:.3f}  agreement {agree}/{n} = {agree / n:.3f} (95% Wilson {lo:.3f}-{hi:.3f})")
    print(f"  max|logit err| median {st.median(errs):.3f} mean {st.mean(errs):.3f} p95 {sorted(errs)[int(0.95 * len(errs)) - 1]:.3f} max {max(errs):.3f}")
    print(f"  per sentence: rounds {st.mean(int(r['rounds']) for r in rows):.0f}, "
          f"{st.mean(int(r['bytes']) for r in rows) / 1e9:.3f} GB, {st.mean(float(r['time_s']) for r in rows):.1f} s, "
          f"mean tokens {st.mean(int(r['n_tokens']) for r in rows):.1f}")

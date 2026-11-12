#!/usr/bin/env python3
"""Summarise secure LLM domain-shift CSVs (mpc_llm_domain_shift.py) and out-of-range statistics
(llm_domain_prep.py --stats_csv) as markdown tables.

A secure window counts as collapsed when the output is non-finite, agrees with the plaintext top-1
token on fewer than half the positions, or at least doubles perplexity.

Usage:
    python mpc_llm_summary.py results/mpc/mpc_llm_domain_shift_n64_d1.0_v2.csv [results/mpc/llm_domain_stats_n64.csv ...]
"""

import csv
import sys
from collections import defaultdict

from mpc_cost_model import estimate


def f(r, k, default=float("nan")):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return default


def garbage(r):
    """Numerically broken output (overflow or divergence), as opposed to a degraded distribution."""
    return r["finite"] != "True" or f(r, "mean_kl_vs_plain") > 100


def collapsed(r):
    return (r["finite"] != "True" or f(r, "top1_agree_vs_plain") < 0.5
            or f(r, "ppl_secure") >= 2 * f(r, "ppl_plain"))


def secure_tables(path, rows):
    print(f"== {path} (N = {rows[0]['seq_len']}, calibrated on {rows[0]['calib']}, delta {rows[0]['delta_nats']} nats)\n")
    print("| domain | window | bits | variant | max excess | rows diverge | rows short | rows underflow16 "
          "| top-1 agree | KL | ppl plain → secure | collapsed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['domain']} | {r['kind']} | {r['precision_bits']} | {r['variant']} | {f(r, 'max_excess_nats'):+.2f} "
              f"| {100 * f(r, 'frac_rows_nr_diverge'):.2f}% | {100 * f(r, 'frac_rows_nr_short'):.1f}% "
              f"| {100 * f(r, 'frac_rows_underflow16'):.1f}% | {f(r, 'top1_agree_vs_plain'):.3f} "
              f"| {f(r, 'mean_kl_vs_plain'):.2e} | {f(r, 'ppl_plain'):.3g} → {f(r, 'ppl_secure'):.3g} "
              f"| {'**yes**' if collapsed(r) else 'no'} |")

    agg = defaultdict(list)
    for r in rows:
        agg[(r["precision_bits"], r["variant"], r["domain"])].append(r)
    print("\n| bits | variant | domain | windows | collapsed | of which garbage (KL > 100) | mean top-1 agree | median KL |")
    print("|---|---|---|---|---|---|---|---|")
    for (bits, var, dom), rs in sorted(agg.items()):
        kls = sorted(f(r, "mean_kl_vs_plain") for r in rs)
        print(f"| {bits} | {var} | {dom} | {len(rs)} | {sum(collapsed(r) for r in rs)} | {sum(garbage(r) for r in rs)} "
              f"| {sum(f(r, 'top1_agree_vs_plain') for r in rs) / len(rs):.3f} | {kls[len(kls) // 2]:.2e} |")

    cost = defaultdict(list)
    for r in rows:
        cost[(r["precision_bits"], r["variant"])].append(r)
    print("\n| bits | variant | rounds | GB | compute s | LAN s | WAN s |")
    print("|---|---|---|---|---|---|---|")
    for (bits, var), rs in sorted(cost.items()):
        rounds = sum(int(r["rounds"]) for r in rs) / len(rs)
        nbytes = sum(f(r, "bytes") for r in rs) / len(rs)
        t = sum(f(r, "time_s") - f(r, "comm_time_s", 0.0) for r in rs) / len(rs)
        est = estimate(rounds, nbytes, t)
        print(f"| {bits} | {var} | {rounds:.0f} | {nbytes / 1e9:.2f} | {t:.1f} | {est['LAN']:.1f} | {est['WAN']:.1f} |")
    print()


def stats_table(path, rows):
    print(f"== {path} (N = {rows[0]['seq_len']}, delta {rows[0]['delta']} nats)\n")
    print("| domain | constant per | windows | windows with a diverging row | rows diverge | rows short "
          "| rows underflow 16 / 20 bit | max excess median / max |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['domain']} | {r['granularity']} | {r['windows']} | {100 * f(r, 'frac_windows_any_diverge'):.0f}% "
              f"| {100 * f(r, 'mean_frac_rows_nr_diverge'):.2f}% | {100 * f(r, 'mean_frac_rows_nr_short'):.1f}% "
              f"| {100 * f(r, 'mean_frac_rows_underflow16'):.1f}% / {100 * f(r, 'mean_frac_rows_underflow20'):.1f}% "
              f"| {f(r, 'median_max_excess_nats'):+.2f} / {f(r, 'max_max_excess_nats'):+.2f} |")
    print()


def main():
    for path in sys.argv[1:]:
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        (stats_table if "granularity" in rows[0] and "windows" in rows[0] else secure_tables)(path, rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Network-time estimates for the secure-inference CSVs, with SHAFT's accounting.

SHAFT (NDSS 2025) reports running time as computation + communication/bandwidth + rounds * latency,
so results do not depend on the machine's loopback. This applies that model to our measured
rounds and bytes for the same two settings SHAFT uses:

  LAN  3 Gbps, 4 ms latency
  WAN  400 Mbps, 40 ms latency

Computation time is taken from the CSV's time column when given, otherwise 0 (network-only).

Usage:
    python mpc_cost_model.py results/mpc/mpc_softmax_mn5.csv [results/mpc/mpc_vit_private_mn5.csv ...]
"""

import csv
import sys

NETWORKS = {"LAN": (3e9, 0.004), "WAN": (400e6, 0.040)}


def estimate(rounds, nbytes, compute_s=0.0):
    return {name: compute_s + 8.0 * nbytes / bw + rounds * lat for name, (bw, lat) in NETWORKS.items()}


def main():
    for path in sys.argv[1:]:
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        rk = "total_rounds" if "total_rounds" in rows[0] else "rounds"
        bk = "total_bytes" if "total_bytes" in rows[0] else "bytes"
        tk = "total_time_s" if "total_time_s" in rows[0] else ("time_s" if "time_s" in rows[0] else None)
        keys = [k for k in ("model", "variant", "softmax_method", "regime", "n", "seq_len", "precision_bits")
                if k in rows[0]]
        print(f"== {path}")
        print("  " + "  ".join(f"{k:>12}" for k in keys) + f"  {'rounds':>7} {'MB':>9} {'LAN s':>8} {'WAN s':>8}")
        for r in rows:
            est = estimate(int(r[rk]), float(r[bk]), float(r[tk]) if tk else 0.0)
            print("  " + "  ".join(f"{r[k]:>12}" for k in keys)
                  + f"  {int(r[rk]):>7} {float(r[bk]) / 1e6:>9.2f} {est['LAN']:>8.2f} {est['WAN']:>8.2f}")


if __name__ == "__main__":
    main()

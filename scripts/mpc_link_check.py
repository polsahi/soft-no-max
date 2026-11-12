#!/usr/bin/env python3
"""Check the network-time model against a real two-machine run.

Joins the two-node CSV (parties on different MN5 nodes, link measured by --net_probe) with the
single-node reference run from the same job (same threads per party), and compares

  predicted = compute + 8 * bytes / measured bandwidth + rounds * measured round trip

with the measured two-node time. If they agree, the LAN/WAN estimates in mpc_cost_model.py (same
formula with SHAFT's link parameters) rest on a validated model rather than on assumption.

compute is the single-node time minus CrypTen's own communication-time counter when the CSV has
it (comm_time_s). Older CSVs lack it; then the single-node total is used, which still contains the
loopback communication, so the prediction double-counts that part and is an upper bound.

Usage:
    python mpc_link_check.py results/mpc/mpc_multinode_bert_base.csv results/mpc/mpc_multinode_bert_base_loopback.csv
"""

import csv
import sys

from mpc_cost_model import estimate


def main():
    two, one = (list(csv.DictReader(open(p))) for p in sys.argv[1:3])
    keys = [k for k in ("arch", "variant", "softmax_method", "layers", "hidden", "seq_len", "n") if k in two[0]]
    ref = {tuple(r[k] for k in keys): r for r in one}
    has_comm = "comm_time_s" in one[0]
    print(f"link: {float(two[0]['link_gbps']):.2f} Gbps, round trip {float(two[0]['link_rtt_ms']):.3f} ms; "
          f"compute = {'single-node total - comm time' if has_comm else 'single-node total (upper bound)'}")
    print(f"{'config':44s} {'rounds':>7} {'GB':>7} {'1-node s':>9} {'compute':>8} {'2n comm':>8} {'predicted':>9}"
          f" {'measured':>9} {'err':>7} {'LAN s':>8} {'WAN s':>8}")
    for r in two:
        k = tuple(r[c] for c in keys)
        if k not in ref:
            continue
        rounds, nbytes = int(r["total_rounds"]), float(r["total_bytes"])
        base = float(ref[k]["total_time_s"])
        compute = base - float(ref[k]["comm_time_s"]) if has_comm else base
        comm2 = float(r["comm_time_s"]) if "comm_time_s" in r else float("nan")
        bw, rtt = float(r["link_gbps"]) * 1e9, float(r["link_rtt_ms"]) / 1e3
        pred = compute + 8.0 * nbytes / bw + rounds * rtt
        meas = float(r["total_time_s"])
        est = estimate(rounds, nbytes, compute)
        print(f"{' '.join(k):44s} {rounds:>7} {nbytes / 1e9:>7.2f} {base:>9.1f} {compute:>8.1f} {comm2:>8.1f}"
              f" {pred:>9.1f} {meas:>9.1f} {100 * (meas - pred) / pred:>+6.1f}% {est['LAN']:>8.1f} {est['WAN']:>8.1f}")


if __name__ == "__main__":
    main()

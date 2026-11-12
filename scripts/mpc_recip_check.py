#!/usr/bin/env python3
"""Working range of CrypTen's secure reciprocal and exp at 16 bits (2 parties, default iterations).

Run from SHAFT's examples/unit-test directory: python mpc_recip_check.py
See MPC_REFRAME.md section 4g' caveat for the measured table.
"""
import math, os, sys
import torch
sys.path.insert(0, os.getcwd())

def main():
    import crypten
    from crypten.config import cfg
    crypten.init()
    rank = crypten.comm.get().get_rank()
    s = torch.tensor([math.exp(-11), math.exp(-8), math.exp(-7.5), math.exp(-6), 1.0, 64.0, 200.0, 250.0, 256.0, 257.0, 300.0, 600.0])
    x = torch.tensor([-50.0, -8.0, 0.0, 3.0, 5.0, 7.0, 9.0, 12.0])
    reps = 200
    with cfg.temp_override({"functions.reciprocal_all_pos": True}):
        es = crypten.cryptensor(s.repeat(reps, 1), src=0)
        rec = es.reciprocal().get_plain_text()
    init = (3 * (1 - 2 * crypten.cryptensor(s, src=0)).exp() + 0.003).get_plain_text()
    ex = crypten.cryptensor(x.repeat(reps, 1), src=0).exp().get_plain_text()
    if rank == 0:
        print("reciprocal, 16-bit, 10 NR iters, 200 reps each: s | true 1/s | median secure | frac within 1% | secure init guess")
        for j in range(len(s)):
            col = rec[:, j]
            ok = ((col * s[j] - 1).abs() < 0.01).float().mean()
            print(f"  s={float(s[j]):10.4g} ln={math.log(float(s[j])):+6.2f} | {1/float(s[j]):10.4g} | {float(col.median()):12.4g} | {float(ok):.3f} | {float(init[j]):.4g}")
        print("exp (limit approx, 8 squarings), 200 reps: x | true | median secure | max rel err")
        for j in range(len(x)):
            col = ex[:, j]
            t = math.exp(float(x[j]))
            print(f"  x={float(x[j]):+6.1f} | {t:10.4g} | {float(col.median()):10.4g} | {float(((col - t).abs() / max(t, 1e-12)).max()):.3g}")

if __name__ == "__main__":
    from multiprocess_launcher import MultiProcessLauncher
    l = MultiProcessLauncher(2, main); l.start(); l.join(); l.terminate()

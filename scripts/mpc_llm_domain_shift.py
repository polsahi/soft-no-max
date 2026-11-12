#!/usr/bin/env python3
"""Point 2: does a calibrated (NEXUS-style) softmax shift survive secure LLM inference under domain shift?

SmolLM2-135M runs end to end in 2-party CrypTen (SHAFT's fork) on windows prepared by
llm_domain_prep.py, with two attention softmaxes:

  crypten_max   secure maximum + exp + reciprocal (CrypTen), shift-invariant reference
  emp_const     shift by the per-layer constant calibrated on Wikitext (max logit + delta), no
                maximum: what NEXUS does with an empirically chosen constant

and the secure next-token distribution is compared with the plaintext model's.

Scope and simplifications, stated in the output:
- Weights are public (plaintext) to both parties; only the activations are secret. Weight privacy
  changes cost, not the numerics this experiment measures.
- The embedding lookup is done by the client in the clear and its output encrypted; the LM head is
  applied to the decrypted final hidden states. Everything between (all decoder layers) is secure.
- RMSNorm divides by a public per-norm, per-position scale (calibration mean square) before the
  secure Newton inverse square root, so the iteration stays in its convergent range.
- The causal mask sends masked logits to MASK_VALUE with a public affine map (no -inf in fixed point).

Before the secure runs, the same forward runs in plaintext float (exact softmax, same mask) and its
KL to the Hugging Face log-probabilities is printed, which checks this re-implementation.

Usage (from SHAFT's examples/unit-test directory):
    python mpc_llm_domain_shift.py --prep llm_prep.pt --variants crypten_max,emp_const \
        --precision_bits 16,20 --out results/mpc/mpc_llm_domain_shift.csv
"""

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("SHAFT_UNIT_TEST_DIR", os.getcwd()))

_ap = argparse.ArgumentParser()
_ap.add_argument("--prep", default="llm_prep.pt")
_ap.add_argument("--variants", default="crypten_max,emp_const")
_ap.add_argument("--precision_bits", default="16,20")
_ap.add_argument("--max_windows", type=int, default=0)
_ap.add_argument("--plain_only", action="store_true")
_ap.add_argument("--out", default="results/mpc/mpc_llm_domain_shift.csv")
ARGS = _ap.parse_known_args()[0]

MASK_VALUE = -50.0


def rope_theta(cfg):
    if cfg.get("rope_theta") is not None:
        return float(cfg["rope_theta"])
    return float(cfg["rope_parameters"]["rope_theta"])


def rope_tables(N, dh, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dh, 2, dtype=torch.float64) / dh))
    freqs = torch.outer(torch.arange(N, dtype=torch.float64), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float(), emb.sin().float()          # N, dh


def parse_emp(variant):
    """emp_<layer|head|row>[_nr<iters>][_eps<k>]: constant granularity, optional Newton iteration count,
    optional public floor e^-k added to every row sum before the reciprocal (keeps 1/sum <= e^k, so the
    iteration cannot overflow 2-party truncation). emp_const is the per-layer constant with defaults."""
    if variant == "emp_const":
        return "layer", 0, 0.0
    parts = variant.split("_")
    nr, eps = 0, 0.0
    for p in parts[2:]:
        if p.startswith("nr"):
            nr = int(p[2:])
        elif p.startswith("eps"):
            eps = math.exp(-float(p[3:]))
    return parts[1], nr, eps


def shift_for(m_star, gran, L, N):
    """Public shift for layer L, broadcastable against logits [1, heads, N, N]."""
    if not isinstance(m_star, dict):                    # older prep files: per-layer tensor
        m_star = {"layer": m_star}
    m = m_star[gran][L]
    if gran == "layer":
        return float(m)
    if gran == "head":
        return m.float()[:, None, None]
    return m.float()[:, :N, None]


def _plain(t):
    return isinstance(t, torch.Tensor)


def _cat(xs, dim):
    if _plain(xs[0]):
        return torch.cat(xs, dim)
    import crypten
    return crypten.cat(xs, dim=dim)


def forward(x, sd, cfg, m_star, scales, variant, record=None):
    """x: input embeddings [1, N, D], encrypted or plaintext. Returns final hidden states (same kind).
    In plaintext, record (a dict) receives each layer's masked row maxima [heads, N]."""
    D = cfg["hidden_size"]; H = cfg["num_attention_heads"]; KV = cfg["num_key_value_heads"]
    dh = cfg.get("head_dim") or D // H; rep = H // KV; eps = cfg["rms_norm_eps"]
    N = x.size(1)
    cos, sin = rope_tables(N, dh, rope_theta(cfg))
    keep = torch.tril(torch.ones(N, N))
    plain = _plain(x)
    si = 0

    def rmsnorm(t, weight, scale):
        s = scale[:N].float()[:, None]                   # N, 1: public preconditioning
        u = t * s.rsqrt()
        v = u.square().mean(dim=-1, keepdim=True) + (eps / s)
        inv = v.rsqrt() if plain else v.inv_sqrt()
        return u * inv * weight

    def lin(t, name):
        return t.matmul(sd[name].t())

    def rotate_half(t):
        half = t.size(-1) // 2
        return _cat([-t[..., half:], t[..., :half]], -1)

    h = x
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        y = rmsnorm(h, sd[p + "input_layernorm.weight"], scales[si]); si += 1
        q = lin(y, p + "self_attn.q_proj.weight").reshape(1, N, H, dh).permute(0, 2, 1, 3)
        k = lin(y, p + "self_attn.k_proj.weight").reshape(1, N, KV, dh).permute(0, 2, 1, 3)
        v = lin(y, p + "self_attn.v_proj.weight").reshape(1, N, KV, dh).permute(0, 2, 1, 3)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # repeat_kv: each key/value head serves `rep` consecutive query heads
        k = _cat([k[:, j:j + 1] for j in range(KV) for _ in range(rep)], 1)
        v = _cat([v[:, j:j + 1] for j in range(KV) for _ in range(rep)], 1)
        logits = q.matmul(k.transpose(-1, -2)) * (1.0 / math.sqrt(dh))
        logits = logits * keep + MASK_VALUE * (1 - keep)
        if plain:
            if record is not None:
                record[L] = logits.masked_fill(keep == 0, float("-inf")).amax(-1)[0]
            attn = logits.softmax(-1)
        elif variant == "crypten_max":
            from crypten.config import cfg as ccfg
            with ccfg.temp_override({"functions.softmax_method": "reciprocal"}):
                attn = logits.softmax(-1)
        elif variant.startswith("emp_"):
            from crypten.config import cfg as ccfg
            gran, nr, eps = parse_emp(variant)
            num = (logits - shift_for(m_star, gran, L, N)).exp()
            den = num.sum(-1, keepdim=True)
            if eps:
                den = den + eps
            override = {"functions.reciprocal_all_pos": True}
            if nr:
                override["functions.reciprocal_nr_iters"] = nr
            with ccfg.temp_override(override):
                attn = num * den.reciprocal()
        else:
            raise ValueError(variant)
        o = attn.matmul(v).permute(0, 2, 1, 3).reshape(1, N, H * dh)
        h = h + lin(o, p + "self_attn.o_proj.weight")
        y = rmsnorm(h, sd[p + "post_attention_layernorm.weight"], scales[si]); si += 1
        g = lin(y, p + "mlp.gate_proj.weight")
        g = F.silu(g) if plain else g.silu()
        h = h + lin(g * lin(y, p + "mlp.up_proj.weight"), p + "mlp.down_proj.weight")
    return rmsnorm(h, sd["model.norm.weight"], scales[si])


def compare(hid, emb_w, ref_lp, ids):
    finite = bool(torch.isfinite(hid).all())
    if not finite:
        hid = torch.nan_to_num(hid, nan=0.0, posinf=1e4, neginf=-1e4)
    lp = torch.log_softmax(hid.double() @ emb_w.double().t(), -1)
    top1 = float((lp.argmax(-1) == ref_lp.argmax(-1)).double().mean())
    kl = float((ref_lp.exp() * (ref_lp - lp)).sum(-1).mean())
    tgt = ids[1:, None]
    nll_ref = float(-ref_lp[:-1].gather(1, tgt).mean())
    nll = float(-lp[:-1].gather(1, tgt).mean())
    return {"finite": finite, "top1_agree_vs_plain": top1, "mean_kl_vs_plain": kl,
            "ppl_plain": math.exp(nll_ref), "ppl_secure": math.exp(min(nll, 700.0))}


def load_prep(path):
    prep = torch.load(path, map_location="cpu")
    sd = prep["state_dict"]
    emb_w = sd["lm_head.weight"] if "lm_head.weight" in sd else sd["model.embed_tokens.weight"]
    windows = prep["windows"][: ARGS.max_windows] if ARGS.max_windows else prep["windows"]
    return prep, sd, emb_w, windows


def plain_check():
    prep, sd, emb_w, windows = load_prep(ARGS.prep)
    for w in windows:
        with torch.no_grad():
            hid = forward(w["embeddings"][None].float(), sd, prep["config"], prep["m_star"], prep["rms_scales"], None)[0]
        r = compare(hid, emb_w, w["logprobs"].double(), w["ids"])
        print(f"[plain check {w['domain']:13s} {w['kind']:6s}] top1 agree {r['top1_agree_vs_plain']:.3f} "
              f"KL {r['mean_kl_vs_plain']:.2e} ppl HF {r['ppl_plain']:.2f} re-impl {r['ppl_secure']:.2f}", flush=True)


def main():
    import crypten
    from crypten.config import cfg as ccfg

    crypten.init()
    rank = crypten.comm.get().get_rank()
    prep, sd, emb_w, windows = load_prep(ARGS.prep)
    cfg, m_star, scales = prep["config"], prep["m_star"], prep["rms_scales"]
    rows = []

    for w in windows:
        ids = w["ids"]
        ref_lp = w["logprobs"].double()
        for prec in [int(p) for p in ARGS.precision_bits.split(",")]:
            for variant in ARGS.variants.split(","):
                with ccfg.temp_override({"encoder.precision_bits": prec}):
                    x = crypten.cryptensor(w["embeddings"][None].float(), src=1)
                    crypten.reset_communication_stats()
                    t0 = time.time()
                    hid = forward(x, sd, cfg, m_star, scales, variant).get_plain_text()[0]
                    dt = time.time() - t0
                    st = crypten.get_communication_stats()
                r = compare(hid, emb_w, ref_lp, ids)
                gran, nr, eps = parse_emp(variant) if variant.startswith("emp_") else (prep.get("select", "layer"), 0, 0.0)
                ws = w["stats"][gran] if "stats" in w else {
                    "max_excess_nats": w["max_excess_nats"], "frac_rows_nr_diverge": float("nan"),
                    "frac_rows_nr_short": float("nan"), "frac_rows_underflow16": w["frac_rows_shortfall_gt_11nats"],
                    "frac_rows_underflow20": float("nan")}
                row = {"domain": w["domain"], "kind": w["kind"], "seq_len": ids.numel(), "precision_bits": prec,
                       "variant": variant, "granularity": gran if variant.startswith("emp_") else "",
                       "nr_iters": nr or 10, "delta_nats": float(prep["delta"]), "calib": prep["calib"],
                       "select": prep.get("select", "layer"), **ws, **r, "rounds": st["rounds"], "bytes": st["bytes"], "time_s": dt,
                       "comm_time_s": float(st.get("time", 0.0))}
                rows.append(row)
                if rank == 0:
                    print(f"[{w['domain']:13s} {w['kind']:6s} excess {ws['max_excess_nats']:+6.2f} prec={prec} "
                          f"{variant:13s}] top1 agree {r['top1_agree_vs_plain']:.3f} KL {r['mean_kl_vs_plain']:.3e} "
                          f"ppl plain {r['ppl_plain']:.2f} secure {r['ppl_secure']:.2f} finite {r['finite']} | "
                          f"{st['rounds']} rd {st['bytes'] / 1e9:.2f} GB {dt:.1f}s", flush=True)
                    os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
                    with open(ARGS.out, "w", newline="") as f:
                        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        wr.writeheader()
                        wr.writerows(rows)
    if rank == 0:
        print(f"[done] {ARGS.out}", flush=True)


if __name__ == "__main__":
    plain_check()
    if not ARGS.plain_only:
        from multiprocess_launcher import MultiProcessLauncher
        launcher = MultiProcessLauncher(2, main)
        launcher.start()
        launcher.join()
        launcher.terminate()

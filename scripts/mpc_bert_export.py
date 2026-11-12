#!/usr/bin/env python3
"""Export a fine-tuned BERT SST-2 checkpoint (vanilla or scalar QK-norm) for secure inference.

Runs in the HF environment (python 3.12, transformers). The secure environment (CrypTen, torch 2.0.1)
has no transformers, so everything it needs is exported as plain tensors:

  - the validation subset (the same indices for every checkpoint: seeded permutation of SST-2 dev),
  - per example: token ids, the embedding output h0 (embeddings + embedding LayerNorm, computed by the
    client in the clear), the label and the plaintext logits of the HF model (reference),
  - encoder, pooler and classifier weights, and for QK-norm the per-layer gain g = g_max*sigmoid(g_raw).

Usage:
    python mpc_bert_export.py --ckpt ckpt/bert_qknorm_sst2_g30 --variant qknorm --g_max 3.0 \
        --data glue/sst2 --n 200 --out export/bert_qknorm_g30_sst2.pt
"""

import argparse
import os

import torch

from bert_qknorm_glue import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True, choices=["vanilla", "qknorm"])
    ap.add_argument("--g_max", type=float, default=5.5)
    ap.add_argument("--data", default="glue/sst2")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--min_tokens", type=int, default=0, help="keep only validation examples with at least this many tokens (long-input run)")
    ap.add_argument("--task", default="sst2", choices=["sst2", "qnli"], help="text fields: sst2 = sentence; qnli = question, sentence")
    ap.add_argument("--calib", type=int, default=128, help="train sentences used to calibrate public inverse-sqrt scales")
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--target", type=float, default=25.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from datasets import load_from_disk
    k1, k2 = {"sst2": ("sentence", None), "qnli": ("question", "sentence")}[a.task]

    def text(row):
        return (row[k1],) if k2 is None else (row[k1], row[k2])
    from transformers import AutoTokenizer

    ck = torch.load(os.path.join(a.ckpt, "model.pt"), map_location="cpu")
    model = build(a.model, a.variant, ck.get("g_max", a.g_max), 2)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_from_disk(a.data)["validation"]
    gen = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(ds), generator=gen).tolist()
    if a.min_tokens:
        keep = [i for i in perm if len(tok(*text(ds[i]), truncation=True, max_length=a.max_len)["input_ids"]) >= a.min_tokens]
        print(f"[export] {len(keep)} of {len(perm)} validation examples have >= {a.min_tokens} tokens")
        perm = keep
    idx = perm[: a.n]

    # Public scales for CrypTen's Newton inverse square root, which diverges for inputs above ~75: calibrated per
    # layer on TRAIN sentences (never the evaluation ones). QK-norm: sum of squares of a head's q or k vector;
    # LayerNorm: variance of the input row. c2 = max(1, margin * max_calib / target); the secure model divides the
    # input by c2 before the inverse square root and rescales the result, so the output is unchanged.
    tr = load_from_disk(a.data)["train"]
    cidx = torch.randperm(len(tr), generator=torch.Generator().manual_seed(a.seed + 1))[: a.calib].tolist()
    nl = len(model.bert.encoder.layer)
    st = {k: [0.0] * nl for k in ("q", "k", "ln1", "ln2")}
    def hook_proj(li, key):
        def f(mod, inp, out):
            st[key][li] = max(st[key][li], float(out.view(*out.shape[:-1], 12, -1).pow(2).sum(-1).max()))
        return f
    def hook_ln(li, key):
        def f(mod, inp):
            st[key][li] = max(st[key][li], float(inp[0].var(-1, unbiased=False).max()))
        return f
    hs = []
    for li, layer in enumerate(model.bert.encoder.layer):
        hs.append(layer.attention.self.query.register_forward_hook(hook_proj(li, "q")))
        hs.append(layer.attention.self.key.register_forward_hook(hook_proj(li, "k")))
        hs.append(layer.attention.output.LayerNorm.register_forward_pre_hook(hook_ln(li, "ln1")))
        hs.append(layer.output.LayerNorm.register_forward_pre_hook(hook_ln(li, "ln2")))
    with torch.no_grad():
        for i in cidx:
            enc = tok(*text(tr[i]), truncation=True, max_length=a.max_len, return_tensors="pt")
            model(**enc)
    for h in hs:
        h.remove()
    c2f = lambda m: max(1.0, a.margin * m / a.target)
    scales = [{"qk_c2": c2f(max(st["q"][li], st["k"][li])), "ln1_c2": c2f(st["ln1"][li]), "ln2_c2": c2f(st["ln2"][li])}
              for li in range(nl)]
    print("[export] calibration maxima (q, k sumsq; ln1, ln2 var) and chosen scales per layer:")
    for li in range(nl):
        print(f"  L{li}: q {st['q'][li]:.1f} k {st['k'][li]:.1f} ln1 {st['ln1'][li]:.2f} ln2 {st['ln2'][li]:.1f} -> "
              f"qk_c2 {scales[li]['qk_c2']:.1f} ln1_c2 {scales[li]['ln1_c2']:.1f} ln2_c2 {scales[li]['ln2_c2']:.1f}")

    examples = []
    with torch.no_grad():
        for i in idx:
            enc = tok(*text(ds[i]), truncation=True, max_length=a.max_len, return_tensors="pt")
            out = model(**enc).logits[0]
            h0 = model.bert.embeddings(input_ids=enc["input_ids"], token_type_ids=enc["token_type_ids"])[0]
            examples.append({"idx": i, "label": int(ds[i]["label"]), "ids": enc["input_ids"][0],
                             "h0": h0.clone(), "ref_logits": out.clone()})

    def w(t):
        return t.detach().clone()

    layers = []
    for layer in model.bert.encoder.layer:
        att, ao = layer.attention.self, layer.attention.output
        d = {"q": w(att.query.weight), "qb": w(att.query.bias), "k": w(att.key.weight), "kb": w(att.key.bias),
             "v": w(att.value.weight), "vb": w(att.value.bias), "o": w(ao.dense.weight), "ob": w(ao.dense.bias),
             "ln1": w(ao.LayerNorm.weight), "ln1b": w(ao.LayerNorm.bias),
             "f1": w(layer.intermediate.dense.weight), "f1b": w(layer.intermediate.dense.bias),
             "f2": w(layer.output.dense.weight), "f2b": w(layer.output.dense.bias),
             "ln2": w(layer.output.LayerNorm.weight), "ln2b": w(layer.output.LayerNorm.bias)}
        if a.variant == "qknorm":
            d["g"] = w(att.g())
        d.update({k: torch.tensor(v) for k, v in scales[len(layers)].items()})
        layers.append(d)
    W = {"layers": layers, "pool": w(model.bert.pooler.dense.weight), "poolb": w(model.bert.pooler.dense.bias),
         "cls": w(model.classifier.weight), "clsb": w(model.classifier.bias)}

    acc = sum(int(e["ref_logits"].argmax()) == e["label"] for e in examples) / len(examples)
    print(f"[export] {a.variant} n={len(examples)} plaintext subset acc {acc:.4f} "
          f"(full dev acc in ckpt: {ck.get('val_acc')}) g={[round(float(l['g']), 3) for l in layers] if a.variant == 'qknorm' else ''}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"variant": a.variant, "weights": W, "examples": examples, "subset_acc": acc,
                "full_dev_acc": ck.get("val_acc"), "g_max": ck.get("g_max")}, a.out)


if __name__ == "__main__":
    main()

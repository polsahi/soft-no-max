#!/usr/bin/env python3
"""Secure 2-party inference cost of BERT (vanilla vs scalar-gain QK-norm) under SHAFT, per softmax.

Communication cost does not depend on the weight values, so this measures cost with the
architecture alone (random or pretrained weights); accuracy comes from bert_qknorm_glue.py
checkpoints separately. The model is imported with CrypTen's ONNX path, exactly as SHAFT's
examples/text-classification/run_glue_private.py does, and one input of --seq_len tokens is run
per configuration:

  vanilla + reciprocal   CrypTen softmax (secure max + exp + reciprocal)
  vanilla + ode          SHAFT softmax (clip + ODE, t = 16)
  qknorm  + shifted      Soft-no-Max: logits g*cos - g <= 0 in the graph, exp + reciprocal, no max

Usage (from SHAFT's examples/unit-test directory):
    python mpc_bert_cost.py --layers 12 --seq_len 128 --out results/mpc/mpc_bert_cost.csv
"""

import argparse
import csv
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("SHAFT_UNIT_TEST_DIR", os.getcwd()))

_ap = argparse.ArgumentParser()
_ap.add_argument("--layers", type=int, default=12)
_ap.add_argument("--hidden", type=int, default=768)
_ap.add_argument("--heads", type=int, default=12)
_ap.add_argument("--seq_len", type=int, default=128)
_ap.add_argument("--g_max", type=float, default=5.5)
_ap.add_argument("--configs", default="vanilla:reciprocal,vanilla:ode,qknorm:shifted")
_ap.add_argument("--out", default="results/mpc/mpc_bert_cost.csv")
ARGS = _ap.parse_known_args()[0]


def build_model(variant, a):
    from transformers import BertConfig, BertForSequenceClassification
    from bert_qknorm_glue import QKNormSelfAttention
    cfg = BertConfig(hidden_size=a.hidden, num_hidden_layers=a.layers, num_attention_heads=a.heads,
                     intermediate_size=4 * a.hidden, num_labels=2, attn_implementation="eager")
    torch.manual_seed(0)
    model = BertForSequenceClassification(cfg).eval()
    if variant == "qknorm":
        for layer in model.bert.encoder.layer:
            layer.attention.self = QKNormSelfAttention(layer.attention.self, a.g_max)
    return model


def main():
    import crypten as ct
    from crypten.config import cfg

    a = ARGS
    ct.init()
    rank = ct.comm.get().get_rank()
    rows = []
    for spec in a.configs.split(","):
        variant, method = spec.split(":")
        model = build_model(variant, a)
        dummy = torch.zeros(1, a.seq_len, dtype=torch.long)
        with cfg.temp_override({"functions.softmax_method": method}):
            private = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).encrypt()
            ids = ct.cryptensor(torch.randint(1000, 2000, (1, a.seq_len)))
            mask = ct.cryptensor(torch.ones(1, a.seq_len))
            types = ct.cryptensor(torch.zeros(1, a.seq_len))
            ct.reset_communication_stats()
            t0 = time.time()
            with ct.no_grad():
                private(ids, mask, types)
            dt = time.time() - t0
            st = ct.get_communication_stats()
        row = {"variant": variant, "softmax_method": method, "layers": a.layers, "hidden": a.hidden,
               "heads": a.heads, "seq_len": a.seq_len, "rounds": st["rounds"], "bytes": st["bytes"],
               "time_s": dt}
        rows.append(row)
        if rank == 0:
            print(f"[{variant:7s} {method:10s}] L={a.layers} N={a.seq_len}: rounds {st['rounds']} "
                  f"bytes {st['bytes'] / 1e9:.3f} GB time {dt:.1f}s", flush=True)
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[done] {a.out}")


if __name__ == "__main__":
    from multiprocess_launcher import MultiProcessLauncher
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()

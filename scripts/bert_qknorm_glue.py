#!/usr/bin/env python3
"""Fine-tune BERT with scalar-gain QK-norm attention, bounded for exact fixed-point secure softmax.

Question: does a pretrained BERT keep its GLUE accuracy when its attention logits are forced into
a provable range [-g, g] small enough for an exact, comparison-free softmax in 16-bit fixed point?

  vanilla  standard BERT self-attention, q.k / sqrt(d_head)
  qknorm   q and k L2-normalised over the head dimension, logits = g_l * cos(q, k) - g_l, one
           learnable gain per layer capped at g_max (g_l = g_max * sigmoid(raw)). The "- g_l"
           shift is in the graph, so the logits entering softmax are provably in [-2 g_l, 0]:
           the secure softmax is exp + reciprocal with no maximum, and g_l can stay encrypted.

16-bit fixed point resolves e^(-2g) only while 2g <= 16 ln 2 = 11.1 nats, i.e. g <= 5.5.
Optional distillation from a fine-tuned vanilla teacher (--teacher).

Usage:
    python bert_qknorm_glue.py --model bert-base-uncased --task sst2 --data glue/sst2 \
        --variant qknorm --g_max 5.5 --epochs 3 --out results/bert_qknorm_sst2.csv --save ckpt/bert_qk_sst2
"""

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# text fields, number of labels, validation split, GLUE's headline metric
TASKS = {
    "sst2": ("sentence", None, 2, "validation", "acc"),
    "qnli": ("question", "sentence", 2, "validation", "acc"),
    "rte": ("sentence1", "sentence2", 2, "validation", "acc"),
    "mrpc": ("sentence1", "sentence2", 2, "validation", "f1"),
    "cola": ("sentence", None, 2, "validation", "mcc"),
    "mnli": ("premise", "hypothesis", 3, "validation_matched", "acc"),
}


class QKNormSelfAttention(nn.Module):
    """Drop-in replacement for transformers' BertSelfAttention (eager path)."""

    def __init__(self, orig, g_max):
        super().__init__()
        self.num_attention_heads = orig.num_attention_heads
        self.attention_head_size = orig.attention_head_size
        self.all_head_size = orig.all_head_size
        self.query, self.key, self.value = orig.query, orig.key, orig.value
        self.dropout = orig.dropout
        self.g_max = g_max
        # start at g = sqrt(d_head) capped to g_max, the usual 1/sqrt(d) scale for unit-norm q, k
        g0 = min(math.sqrt(self.attention_head_size), 0.95 * g_max)
        self.g_raw = nn.Parameter(torch.tensor(math.log(g0 / (g_max - g0))))

    def g(self):
        return self.g_max * torch.sigmoid(self.g_raw)

    def _shape(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

    @staticmethod
    def _l2norm(x):
        # x / ||x|| written with Pow/ReduceSum/Sqrt/Div, which CrypTen's ONNX importer supports
        # (F.normalize exports as ReduceL2, which it does not)
        return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + 1e-12)

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        q = self._l2norm(self._shape(self.query(hidden_states)))
        k = self._l2norm(self._shape(self.key(hidden_states)))
        v = self._shape(self.value(hidden_states))
        g = self.g()
        logits = g * (q @ k.transpose(-1, -2)) - g          # provably in [-2g, 0]
        if attention_mask is not None:
            logits = logits + attention_mask
        probs = self.dropout(torch.softmax(logits, dim=-1))
        ctx = (probs @ v).permute(0, 2, 1, 3).reshape(hidden_states.size(0), -1, self.all_head_size)
        # transformers 5 unpacks (attn_output, attn_weights) unconditionally
        return ctx, (probs if kwargs.get("output_attentions") else None)


def build(model_name, variant, g_max, num_labels):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, attn_implementation="eager")
    if variant == "qknorm":
        for layer in model.bert.encoder.layer:
            layer.attention.self = QKNormSelfAttention(layer.attention.self, g_max)
    return model


def evaluate(model, loader, dev):
    """Accuracy, plus F1 of class 1 and Matthews correlation (meaningful for binary tasks)."""
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(dev) for k, v in b.items()}
            labels = b.pop("labels")
            preds.append(model(**b).logits.argmax(-1).cpu()); gold.append(labels.cpu())
    p, y = torch.cat(preds), torch.cat(gold)
    acc = float((p == y).double().mean())
    tp = float(((p == 1) & (y == 1)).sum()); tn = float(((p == 0) & (y == 0)).sum())
    fp = float(((p == 1) & (y == 0)).sum()); fn = float(((p == 0) & (y == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"acc": acc, "f1": f1, "mcc": mcc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--task", default="sst2", choices=list(TASKS))
    ap.add_argument("--data", required=True, help="datasets.save_to_disk directory")
    ap.add_argument("--variant", default="qknorm", choices=["vanilla", "qknorm"])
    ap.add_argument("--g_max", type=float, default=5.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--g_lr", type=float, default=0.0, help="learning rate for QK-norm gains (0: same as --lr)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--teacher", default="", help="fine-tuned vanilla checkpoint dir for distillation")
    ap.add_argument("--kd_alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bert_qknorm.csv")
    ap.add_argument("--save", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N train and validation examples")
    a = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_from_disk(a.data)
    k1, k2, num_labels, val_split, metric = TASKS[a.task]
    if a.limit:
        ds = type(ds)({s: ds[s].select(range(min(a.limit, len(ds[s])))) for s in ("train", val_split)})
    enc = ds.map(lambda e: tok(e[k1], e[k2], truncation=True, max_length=a.max_len) if k2
                 else tok(e[k1], truncation=True, max_length=a.max_len), batched=True)
    keep = ["input_ids", "attention_mask", "token_type_ids", "label"]
    enc = enc.remove_columns([c for c in enc["train"].column_names if c not in keep]).rename_column("label", "labels")
    coll = DataCollatorWithPadding(tok)
    train = torch.utils.data.DataLoader(enc["train"], batch_size=a.bs, shuffle=True, collate_fn=coll)
    val = torch.utils.data.DataLoader(enc[val_split], batch_size=128, collate_fn=coll)

    model = build(a.model, a.variant, a.g_max, num_labels).to(dev)
    teacher = None
    if a.teacher:
        from transformers import AutoModelForSequenceClassification
        teacher = AutoModelForSequenceClassification.from_pretrained(a.teacher).to(dev).eval()
    # The QK-norm gains get their own learning rate: at the fine-tuning lr (2e-5) a gain barely
    # moves from its initial value, so the architecture never adapts its temperature.
    g_params = [p for n, p in model.named_parameters() if n.endswith("g_raw")]
    other = [p for n, p in model.named_parameters() if not n.endswith("g_raw")]
    groups = [{"params": other, "lr": a.lr, "weight_decay": 0.01}]
    if g_params:
        groups.append({"params": g_params, "lr": a.g_lr if a.g_lr > 0 else a.lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    steps = a.epochs * len(train)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for b in train:
            b = {k: v.to(dev) for k, v in b.items()}
            labels = b.pop("labels")
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(**b).logits
            loss = F.cross_entropy(logits.float(), labels)
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(**b).logits.float()
                loss = (1 - a.kd_alpha) * loss + a.kd_alpha * F.kl_div(
                    F.log_softmax(logits.float(), -1), F.softmax(tl, -1), reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = evaluate(model, val, dev)
        print(f"[{a.task} {a.variant}] epoch {ep + 1}/{a.epochs} val acc {m['acc']:.4f} {metric} {m[metric]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    gs = [float(l.attention.self.g().detach()) for l in model.bert.encoder.layer] if a.variant == "qknorm" else []
    row = {"task": a.task, "variant": a.variant, "g_max": a.g_max if a.variant == "qknorm" else "",
           "teacher": bool(a.teacher), "epochs": a.epochs, "lr": a.lr, "seed": a.seed, "val_acc": m["acc"],
           "metric": metric, "metric_value": m[metric], "f1": m["f1"], "mcc": m["mcc"],
           "g_per_layer": ";".join(f"{g:.3f}" for g in gs), "max_g": max(gs) if gs else "",
           "logit_width_nats": 2 * max(gs) if gs else "", "train_s": time.time() - t0}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(row, flush=True)
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "variant": a.variant, "g_max": a.g_max,
                    "model": a.model, "task": a.task, "val_acc": m["acc"], "num_labels": num_labels},
                   os.path.join(a.save, "model.pt"))


if __name__ == "__main__":
    main()

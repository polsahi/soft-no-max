# Soft-no-Max: public-shift softmax for secure inference of bounded-attention transformers

Code, job scripts and per-sentence results for the paper in `paper/` ("A Public-Shift Softmax for Secure Inference of
Bounded-Attention Transformers: Cost, Failure Modes, and the Price of the Guarantee").

A public constant shift replaces the secure row maximum in softmax when the model bounds its attention logits by
construction (scalar-gain QK-norm, soft-capping). The scripts measure the cost of the resulting two-party protocol in
CrypTen (SHAFT's fork), the fixed-point conditions it needs, and the accuracy cost of the bound.

## Layout

- `paper/` – LaTeX source, bibliography, figures and the compiled PDF.
- `scripts/` – Python scripts.
  - `mpc_encoder_cost.py`: secure cost (rounds, bytes, time) of BERT-base / GPT-2-shaped encoders per softmax variant.
  - `mpc_bert_export.py`, `mpc_bert_sst2.py`, `mpc_bert_sst2_summary.py`: export a fine-tuned BERT classifier and run it in
    two-party inference on SST-2 or QNLI.
  - `mpc_recip_check.py`, `mpc_bound_rows.py`, `mpc_overshoot_sweep.py`: reciprocal window and worst-case rows.
  - `mpc_llm_domain_shift.py`: SmolLM2-135M calibrated-constant experiment.
  - `bert_qknorm_glue.py`, `vit_qknorm_cifar.py`: QK-norm training (BERT on GLUE, ViT on CIFAR-10).
- `slurm/` – the job scripts used on MareNostrum 5 (account, queue and module names are site specific).
- `results/` – CSV outputs: `mpc/` (cost, secure inference, diagnostics) and `training/` (accuracy).

## Requirements

- Secure runs: Python 3.11, torch 2.0.1 and the SHAFT fork of CrypTen (Kei and Chow, NDSS 2025), run from
  `SHAFT/examples/unit-test`. This environment has no `transformers`; export plain tensors first with `mpc_bert_export.py`.
- Training and export: Python 3.12, torch, transformers, datasets (offline copies of `bert-base-uncased` and GLUE).

The fine-tuned checkpoints (about 4 GB) are not in this repository and are available on request.

## Notes

- `mpc_encoder_cost.py --configs` takes `variant:method` pairs, for example
  `vanilla:crypten_max,vanilla:shaft_ode,qknorm:snm_shifted`.
- The reciprocal change used for the fairness control (`functions.softmax_shift_rescale` in CrypTen's `reciprocal` softmax
  branch) is described in the paper; the default value 1.0 leaves CrypTen unchanged.

## License

- Code and job scripts (`scripts/`, `slurm/`): Apache License 2.0, see `LICENSE`.
- Paper, figures and result files (`paper/`, `results/`): CC BY 4.0, see `LICENSE-paper`.
- The repository does not include CrypTen or SHAFT, which have their own licenses.

# OSSA

> **Open Sparse Subquadratic Attention.** Take a frozen Hugging Face LM,
> attach a tiny content router that picks the top-K keys per query,
> swap dense attention for sparse — without retraining the model.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: research preview](https://img.shields.io/badge/status-research_preview-orange.svg)](#honest-scope)

---

## TL;DR

Frozen attention is **already sparse**: in `Qwen/Qwen2.5-1.5B-Instruct`,
top-32 of 1024 keys carries **~92 %** of softmax mass on average. Replacing
dense attention with the oracle top-K of true scores costs **0 %** perplexity
across all 28 layers at any K ≥ 32. So the only thing standing between a
frozen LM and SubQ-style sparse inference is a router that finds those keys
without computing the full Q·K^T.

We trained one such router. **17 408 parameters, 22 minutes on an RTX 3050,
and +2.3 % perplexity at top-K = 64 of 512** on layer 14 of Qwen-1.5B.

---

## Headline numbers

### Frozen Qwen-1.5B is intrinsically sparse

```
Oracle top-K perplexity penalty on all 28 layers, no router:

k        seq=256    seq=512    seq=1024
 8        +3.4 %    +5.4 %     +3.0 %
16        +0.6 %    +2.0 %     +0.9 %
32        −0.9 %    −0.1 %     +0.0 %      ← 3 % of keys, 0 % loss
64        −0.2 %    −0.1 %     +0.0 %
128       +0.0 %    −0.1 %     −0.0 %
```

(`bench/results/sweep_k.json`)

### A 17 k-param content router on layer 14

```
ppl on Qwen-1.5B layer 14, seq=512, K-sweep with one trained router:

k        oracle %     router %     gap
16        −0.10 %      +5.54 %     +0.25
32        −0.03 %      +3.72 %     +0.17
64        −0.00 %      +2.26 %     +0.10
128       +0.00 %      −0.14 %     −0.01     ← router matches oracle
```

(`bench/results/router_sweep.json`)

Router specs:
- `proj_dim=64`, learnable `W_q'`, `W_k'`, position bias
- 17 408 parameters total
- Trained for 1 500 steps, ~22 min on RTX 3050
- Final hold-out recall@64 = **0.705** on 5 unseen prompts

---

## Why this matters

[Subquadratic.ai](https://subq.ai) (May 2026) trained a 12 M-token-context
LM **from scratch** with content-routed sparse attention. They needed
`$29 M` of seed funding and a fresh foundation model to do it.

OSSA proves the same architectural pattern can be **retrofitted onto an
existing frozen model**: the sparsity is already there, you just have to
learn the routing on top of it. No fine-tuning of the LM, no new
foundation model, no $29 M.

What this gives you (when finished):
- **Long context on existing checkpoints.** No retraining Qwen / Llama /
  Mistral to extend their context to 32k+.
- **Compute saving.** O(N·K) attention instead of O(N²) — provided you
  ship a real kernel (see [Roadmap](#roadmap)).
- **A reusable recipe.** The same router approach works on any HF
  causal LM with eager attention.

---

## Method

```
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ Frozen Qwen  │   │ ContentRouter    │   │ Sparse forward  │
│ Q, K, V proj │──▶│ q' · k' + bias   │──▶│ top-K · attn    │
│ (no grad)    │   │ 17 k params      │   │ O(N·K) instead  │
└──────────────┘   └──────────────────┘   │ of O(N²)        │
                                          └─────────────────┘
```

1. **Sparsity probe.** Run frozen LM on long input, dump per-layer
   softmax attention, measure how much mass lives in top-K keys.
   `bench/sparsity.py`
2. **Oracle ceiling.** Replace dense attention with top-K-of-dense-scores
   sparse attention on every layer at once, measure perplexity penalty.
   `bench/sweep_k.py`
3. **Train the router.** A `ContentRouter` projects real `q_proj` /
   `k_proj` of the wrapped layer, scores `q' · k'`, learns to predict
   which keys end up in dense top-K. Loss = BCE on the
   "is this key in dense top-K?" target.
   `bench/content_train.py`
4. **Sparse forward.** Patch the wrapped layer's `forward` to score
   queries with the router, take top-K, attend only to those keys.
   `bench/sparse_forward.py`, `src/ossa/sparse_attention.py`
5. **Sweep / wall-clock.** Grid over K and seq_len, measure both
   perplexity and time. `bench/router_sweep.py`, `bench/wallclock.py`

---

## Quick run

```bash
git clone https://github.com/narelabs/ossa
cd ossa
pip install -e .

# 1. Sparsity probe (~15 s)
python -m ossa.bench.sparsity --seq_len 1024

# 2. Oracle ceiling — every layer patched (~1 min)
python -m ossa.bench.sweep_k --seq_lens 256 512 1024 --ks 8 16 32 64 128

# 3. Train one router on layer 14 (~22 min on RTX 3050)
python -m ossa.bench.content_train --layer 14 --steps 1500

# 4. Router perplexity sweep (~30 s)
python -m ossa.bench.router_sweep --layer 14 \
    --checkpoint checkpoints/content_router_layer14.pt
```

All numbers print to stdout and are written to `bench/results/*.json`.

---

## Honest scope

This is a **research preview**, not a deployable library yet. Read this
before getting excited.

**What works:**
- One layer, one router → published numbers above.
- Algorithmically correct sparse forward (`src/ossa/sparse_attention.py`,
  3 unit tests pass: full top-K matches dense, chunked matches full,
  oracle top-8 cosine > 0.95 to dense).
- The pipeline runs end-to-end on a single RTX 3050.

**What does not yet work:**
1. **Wall-clock speedup.** Pure-PyTorch sparse forward is **slower** than
   dense matmul (BLAS / cuBLAS are too good). On CPU at seq=1024 our
   sparse forward is 0.21–0.99x of dense. A real Triton/CUDA kernel is
   required to convert the algorithmic O(N·K) saving into a wall-clock
   saving. This is on the roadmap and not bluffed about.
2. **All 28 layers with routers.** Headline number is for **layer 14
   only**; other layers still use dense. Training all 28 takes ~14 h on
   RTX 3050 and runs in fixed-cost batches; planned.
3. **Long context.** seq_len = 512 in the headline. The router's
   position bias has size `2 × seq_len`, so a longer-context model
   needs retraining. We have not run seq=4096+ yet.
4. **Dataset.** Perplexity is measured on a stitched seven-paragraph
   reference text, not WikiText. WikiText eval is one of the next steps.

We are publishing now because the **upper bound is striking** (0 % ppl
penalty at 3 % of keys) and the **first router result is honest** (+2.3 %
ppl on one layer). If you find this useful or want to help finish the
list above, open an issue or PR.

---

## Files

```
src/ossa/
  capture.py             Hooked load + per-layer attention dump
  router.py              ContentRouter (17 k params), distillation_loss
  sparse_attention.py    True O(N·K) sparse forward + 3 unit tests
  bench/
    sparsity.py          per-layer mass probe
    sweep_k.py           oracle ceiling vs k & seq_len
    content_train.py     train one router via attention distillation
    multi_layer_train.py train routers on several layers in one run
    router_sweep.py      perplexity sweep with a trained router
    sparse_forward.py    end-to-end patched forward (mask & gather)
    wallclock.py         dense vs sparse timing benchmark

bench/results/           JSON output of every benchmark above
checkpoints/             Trained router .pt files
tests/                   pytest smoke + correctness tests
```

---

## Roadmap

- [ ] Triton kernel for `sparse_attention_forward` so wall-clock catches
      up with the algorithmic O(N·K).
- [ ] Train routers for all 28 Qwen layers, publish full-stack ppl.
- [ ] Re-train at seq_len 2 048 / 4 096; rerun every benchmark.
- [ ] Switch perplexity eval to WikiText-103.
- [ ] Try different base models: Llama-3.2-3B, Mistral-7B.
- [ ] Investigate single-router-shared-across-layers vs per-layer.

---

## Citation

```bibtex
@misc{ossa2026,
  title  = {OSSA: Open Sparse Subquadratic Attention},
  author = {NARE Labs},
  year   = {2026},
  url    = {https://github.com/narelabs/ossa}
}
```

---

## License

[Apache-2.0](LICENSE). NARE Labs, 2026.

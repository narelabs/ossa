"""End-to-end sparse forward + perplexity benchmark.

This is the final question for OSSA. Given a trained router checkpoint
for one layer (and a router-free fallback for the others), how much
perplexity do we lose by replacing dense attention with top-K sparse
attention picked by the router?

If top-K=64 gives ≤5 % perplexity penalty, the project is real.
If it gives 20-50 %, the router architecture isn't strong enough.
If it gives >100 %, sparse retrofit on Qwen-1.5B doesn't work at all.

We run three configurations and compare:

  - dense          : original Qwen attention (baseline)
  - oracle_topk    : keep only the top-K dense scores per query (no router)
                      — this is the upper bound on what any router can do
  - router_topk    : replace dense scores with router predictions, take top-K

The gap dense → oracle_topk tells us **how much the model itself can lose
to sparsity**. The gap oracle_topk → router_topk tells us **how much the
router still has to learn**. The full gap dense → router_topk is the
publishable number.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ossa.capture import load_model
from ossa.router import (
    ContentRouter,
    ContentRouterConfig,
    PerLayerRouter,
    RouterConfig,
)
from ossa.sparse_attention import sparse_attention_forward_chunked


# Long evaluation text. Stitched from many distinct paragraphs so the
# resulting perplexity is meaningful (a single repeated paragraph would
# make the model memorise and the ppl is meaningless ~1).
EVAL_TEXT = (
    "Transformer language models compute attention as a softmax over all keys "
    "for every query. The cost is quadratic in sequence length, which becomes "
    "the bottleneck once the context exceeds a few thousand tokens.\n\n"
    "In May 2026 a small startup called Subquadratic.ai released a model with "
    "twelve million tokens of context. Their architecture relies on content-"
    "based sparse attention with a learned router that selects a small subset "
    "of keys for every query. The router is small relative to the language "
    "model and is trained jointly from scratch.\n\n"
    "Plate tectonics is the theory that Earth's outer shell is divided into "
    "rigid plates that slide over the more fluid asthenosphere. The boundaries "
    "between these plates are the sites of most earthquakes, volcanoes, and "
    "mountain building events on the planet.\n\n"
    "A turbofan engine is a kind of jet engine in which a fan accelerates a "
    "large mass of air around the core. The bypass ratio is the mass flow rate "
    "of air bypassing the core divided by that flowing through the core. High "
    "bypass engines are quieter and more fuel efficient at subsonic speeds.\n\n"
    "In a relational database the join operation combines rows from two tables "
    "based on a related column. The query optimiser chooses an execution plan "
    "by estimating the cost of different join orders and physical operators "
    "such as hash join, sort-merge join, and nested loop join.\n\n"
    "When a star like the Sun runs out of hydrogen in its core it expands into "
    "a red giant, eventually shedding its outer layers as a planetary nebula. "
    "The exposed core remains as a white dwarf, slowly cooling over billions "
    "of years. More massive stars end their lives in supernova explosions "
    "leaving behind neutron stars or black holes.\n\n"
    "Cryptographic hash functions take a variable length input and produce a "
    "fixed length digest. They must be preimage resistant, second preimage "
    "resistant, and collision resistant. SHA-256 produces a 256 bit digest and "
    "is widely used in TLS and digital signatures.\n\n"
)


# ---------------------------------------------------------------------------
# Sparse attention masks
# ---------------------------------------------------------------------------


def _causal_mask(N: int, device: torch.device) -> torch.Tensor:
    """1 where (q, k) is allowed (k <= q), 0 otherwise. Shape (N, N)."""
    return torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))


def _topk_mask_from_scores(
    scores: torch.Tensor, k: int, causal: torch.Tensor
) -> torch.Tensor:
    """Given scores ``(B, H, N, N)`` keep top-k allowed keys per query.

    Returns a boolean mask ``(B, H, N, N)`` with True at kept positions.
    Always keeps the diagonal (q == k) so the attention has at least one
    legal target.
    """
    masked = scores.masked_fill(~causal, float("-inf"))
    # top-k indices per query
    k_eff = min(k, masked.shape[-1])
    top = torch.topk(masked, k=k_eff, dim=-1).indices  # (B, H, N, k)
    keep = torch.zeros_like(masked, dtype=torch.bool)
    keep.scatter_(-1, top, True)
    # Force diagonal so a query can always attend to itself.
    diag_idx = torch.arange(masked.shape[-1], device=scores.device)
    keep[..., diag_idx, diag_idx] = True
    keep &= causal  # respect causality
    return keep


# ---------------------------------------------------------------------------
# Patched attention forward
# ---------------------------------------------------------------------------


@dataclass
class PatchPlan:
    mode: str                          # "dense" | "oracle" | "router"
    k: int = 64
    layers: tuple[int, ...] = ()       # which layers to patch; empty = none
    routers: dict[int, PerLayerRouter] | None = None  # layer_idx -> router
    impl: str = "mask"                 # "mask" (compute full scores then mask)
                                       # | "gather" (true O(N·K) sparse forward)


def _patch_layer(model, plan: PatchPlan):
    """Monkey-patch ``Qwen2Attention.forward`` of the chosen layer(s) to do
    sparse attention. Returns a callable to undo all patches.

    Two implementations:

    * ``impl="mask"`` — compute full ``Q K^T``, mask non-top-K, softmax,
      ``A V``. Same FLOPs as dense, useful only for measuring **quality**.
    * ``impl="gather"`` — pick top-K indices, gather only K keys/values
      per query, compute K dot products and weighted sums. The actual
      sparse forward; matches what a Triton kernel would do on real
      hardware. Used for **wall-clock** comparisons against dense.
    """

    if plan.mode == "dense" or not plan.layers:
        return lambda: None  # no-op

    restores: list = []
    for layer_idx in plan.layers:
        layer_module = model.model.layers[layer_idx].self_attn
        original_forward = layer_module.forward
        captured_idx = layer_idx  # bind in closure

        def make_forward(layer_module=layer_module, captured_idx=captured_idx,
                         original_forward=original_forward):
            def new_forward(hidden_states, attention_mask=None, position_ids=None,
                            past_key_values=None, output_attentions=False,
                            use_cache=False, cache_position=None,
                            position_embeddings=None, **kwargs):
                B, N, _ = hidden_states.shape
                head_dim = layer_module.head_dim
                n_heads = layer_module.config.num_attention_heads
                n_kv_heads = layer_module.config.num_key_value_heads

                q = layer_module.q_proj(hidden_states).view(B, N, n_heads, head_dim).transpose(1, 2)
                k = layer_module.k_proj(hidden_states).view(B, N, n_kv_heads, head_dim).transpose(1, 2)
                v = layer_module.v_proj(hidden_states).view(B, N, n_kv_heads, head_dim).transpose(1, 2)

                if position_embeddings is not None:
                    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
                    cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)

                rep = n_heads // n_kv_heads
                if rep > 1:
                    k = k.repeat_interleave(rep, dim=1)
                    v = v.repeat_interleave(rep, dim=1)

                causal = _causal_mask(N, hidden_states.device)

                # ----- pick top-K indices ---------------------------
                # In ``router`` + ``gather`` mode we never compute the
                # full Q·K^T; the router provides logits directly.
                if plan.mode == "router":
                    assert plan.routers is not None and captured_idx in plan.routers
                    router = plan.routers[captured_idx]
                    with torch.no_grad():
                        if isinstance(router, ContentRouter):
                            router_logits = router.score(q.float(), k.float())
                        else:
                            router_logits = router(q.float())
                    selector = router_logits.to(q.dtype)
                elif plan.mode == "oracle":
                    scale = 1.0 / math.sqrt(head_dim)
                    selector = torch.matmul(q, k.transpose(-2, -1)) * scale
                else:
                    raise ValueError(plan.mode)

                selector = selector.masked_fill(~causal, float("-inf"))
                k_eff = min(plan.k, N)
                top_idx = torch.topk(selector, k=k_eff, dim=-1).indices  # (B, H, N, k)

                if plan.impl == "gather":
                    # True O(N·K) forward: only K dot products per query.
                    out = sparse_attention_forward_chunked(
                        q, k, v, topk_indices=top_idx, chunk_size=64, causal=True,
                    )
                else:
                    # "mask" implementation: compute full scores then keep only top-K.
                    # Same FLOPs as dense; useful for measuring quality only.
                    scale = 1.0 / math.sqrt(head_dim)
                    scores_full = torch.matmul(q, k.transpose(-2, -1)) * scale
                    keep = torch.zeros_like(scores_full, dtype=torch.bool)
                    keep.scatter_(-1, top_idx, True)
                    diag_idx = torch.arange(N, device=q.device)
                    keep[..., diag_idx, diag_idx] = True
                    keep &= causal
                    scores_full = scores_full.masked_fill(~keep, float("-inf"))
                    attn = torch.softmax(scores_full, dim=-1)
                    attn = torch.nan_to_num(attn, nan=0.0)
                    out = torch.matmul(attn, v)

                out = out.transpose(1, 2).contiguous().view(B, N, n_heads * head_dim)
                out = layer_module.o_proj(out)

                if output_attentions:
                    return out, None
                return out, None
            return new_forward

        layer_module.forward = make_forward()
        restores.append((layer_module, original_forward))

    def restore() -> None:
        for lm, orig in restores:
            lm.forward = orig

    return restore


# ---------------------------------------------------------------------------
# Perplexity over sliding windows
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_ppl(model, tokenizer, text: str, *, seq_len: int) -> float:
    """Mean cross-entropy loss over a single window of length ``seq_len``."""
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    ids = enc["input_ids"].to(device)
    if ids.shape[1] < 8:
        raise ValueError("text too short")
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits[:, :-1, :]
    target = ids[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
    )
    return float(torch.exp(loss).item())


def _build_text(seq_len: int, tokenizer) -> str:
    text = EVAL_TEXT
    while len(tokenizer.encode(text)) < seq_len + 16:
        text = text + " " + EVAL_TEXT
    return text


def _load_router(checkpoint: Path, model, layer: int, seq_len: int):
    """Load a router from a checkpoint. Detects ContentRouter vs PerLayerRouter
    by inspecting the saved state dict keys."""
    state = torch.load(checkpoint, map_location=model.device, weights_only=False)
    cfg_dict = state["cfg"]
    state_dict = state["state_dict"]
    layer_module = model.model.layers[layer].self_attn
    head_dim = layer_module.head_dim
    n_heads = layer_module.config.num_attention_heads

    is_content = any(k.startswith("q_proj") or k.startswith("k_proj") for k in state_dict)
    if is_content:
        proj_dim = cfg_dict.get("proj_dim", 64)
        cfg = ContentRouterConfig(
            head_dim=head_dim, n_heads=n_heads, seq_len=seq_len, proj_dim=proj_dim,
        )
        router = ContentRouter(cfg).to(model.device)
    else:
        cfg = RouterConfig(head_dim=head_dim, n_heads=n_heads, seq_len=seq_len)
        router = PerLayerRouter(cfg).to(model.device)
    router.load_state_dict(state_dict)
    router.eval()
    return router


def run(
    *,
    model_name: str,
    layers: tuple[int, ...],
    seq_len: int,
    k: int,
    checkpoint: Optional[Path],
    save: Optional[Path],
    device: Optional[str],
) -> dict:
    started = time.perf_counter()
    print(f"loading {model_name} ...")
    model, tokenizer = load_model(model_name, device=device)
    text = _build_text(seq_len, tokenizer)

    n_tokens = len(tokenizer.encode(text))
    print(f"text tokenized to {n_tokens} tokens (truncated to {seq_len} for eval)")
    print(f"patching layers: {list(layers)}")

    # 1. Dense baseline
    plan = PatchPlan(mode="dense", k=k, layers=())
    restore = _patch_layer(model, plan)
    ppl_dense = compute_ppl(model, tokenizer, text, seq_len=seq_len)
    restore()
    print(f"  dense       ppl = {ppl_dense:.3f}")

    # 2. Oracle top-K (uses true scores, just keeps top-K) on every patched layer
    plan = PatchPlan(mode="oracle", k=k, layers=tuple(layers))
    restore = _patch_layer(model, plan)
    ppl_oracle = compute_ppl(model, tokenizer, text, seq_len=seq_len)
    restore()
    print(f"  oracle k={k:<3d} ppl = {ppl_oracle:.3f}  (delta {(ppl_oracle/ppl_dense - 1):+.1%})")

    # 3. Router top-K (only if checkpoint provided). One checkpoint applied to
    #    one layer; the rest fall back to oracle (better than dense for a fair
    #    "could it work?" comparison if you only trained one router).
    ppl_router: Optional[float] = None
    if checkpoint is not None and len(layers) == 1:
        the_layer = layers[0]
        router = _load_router(checkpoint, model, the_layer, seq_len)
        plan = PatchPlan(mode="router", k=k, layers=(the_layer,), routers={the_layer: router})
        restore = _patch_layer(model, plan)
        ppl_router = compute_ppl(model, tokenizer, text, seq_len=seq_len)
        restore()
        print(
            f"  router k={k:<3d} ppl = {ppl_router:.3f}  "
            f"(delta {(ppl_router/ppl_dense - 1):+.1%})"
        )
    elif checkpoint is not None:
        print("(skip router pass: only single-layer router checkpoint supported here)")

    elapsed = time.perf_counter() - started
    print(f"\nelapsed {elapsed:.1f}s")

    # Verdict
    print()
    print("verdict:")
    delta_oracle = ppl_oracle / ppl_dense - 1
    if delta_oracle <= 0.05:
        print(f"  oracle top-{k} loses {delta_oracle:+.1%} ppl --- attention IS sparse enough.")
    elif delta_oracle <= 0.20:
        print(f"  oracle top-{k} loses {delta_oracle:+.1%} ppl --- bigger k may be needed.")
    else:
        print(f"  oracle top-{k} loses {delta_oracle:+.1%} ppl --- model uses non-top mass at this layer.")

    if ppl_router is not None:
        delta_router = ppl_router / ppl_dense - 1
        gap = ppl_router - ppl_oracle
        if delta_router <= 0.05:
            v = "STRONG"
        elif delta_router <= 0.20:
            v = "MEDIUM"
        else:
            v = "WEAK"
        print(
            f"  router top-{k} loses {delta_router:+.1%} ppl, "
            f"router-vs-oracle gap = {gap:+.3f} -> {v}"
        )

    out = {
        "model": model_name,
        "layers": list(layers),
        "seq_len": seq_len,
        "k": k,
        "ppl_dense": ppl_dense,
        "ppl_oracle": ppl_oracle,
        "ppl_router": ppl_router,
        "delta_oracle": delta_oracle,
        "delta_router": (ppl_router / ppl_dense - 1) if ppl_router is not None else None,
        "elapsed_s": elapsed,
    }
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[14],
                        help="Layer indices to replace; pass several to patch them all.")
    parser.add_argument("--all-layers", action="store_true",
                        help="Patch all 28 transformer layers (overrides --layers).")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Trained router .pt; if omitted only dense + oracle are run.")
    parser.add_argument("--save", type=Path, default=Path("bench/results/sparse_forward.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.all_layers:
        # Detect from a quick load (cheap; will be cached)
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        layers = tuple(range(cfg.num_hidden_layers))
    else:
        layers = tuple(args.layers)

    run(
        model_name=args.model,
        layers=layers,
        seq_len=args.seq_len,
        k=args.k,
        checkpoint=args.checkpoint,
        save=args.save,
        device=args.device,
    )


if __name__ == "__main__":
    main()

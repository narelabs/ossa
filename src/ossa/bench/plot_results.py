"""Generate figures for README from saved JSON results.

Outputs:
  bench/figures/oracle_ceiling.png    -- oracle penalty vs k for each seq_len
  bench/figures/router_vs_oracle.png  -- one trained router on layer 14
  bench/figures/sparsity_per_layer.png -- top-K mass averaged across heads
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_oracle_ceiling(
    sweep_path: Path = Path("bench/results/sweep_k.json"),
    out: Path = Path("bench/figures/oracle_ceiling.png"),
) -> None:
    data = json.loads(sweep_path.read_text())
    rows = data["rows"]
    seqs = sorted({r["seq_len"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for sl in seqs:
        sub = [r for r in rows if r["seq_len"] == sl]
        ks = [r["k"] for r in sub]
        d = [r["delta"] * 100 for r in sub]
        ax.plot(ks, d, marker="o", label=f"seq={sl}")
    ax.set_xscale("log", base=2)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("top-K (log)")
    ax.set_ylabel("perplexity penalty (%)")
    ax.set_title("Oracle ceiling — frozen Qwen-1.5B, all 28 layers patched")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_router_vs_oracle(
    sweep_path: Path = Path("bench/results/router_sweep.json"),
    out: Path = Path("bench/figures/router_vs_oracle.png"),
) -> None:
    data = json.loads(sweep_path.read_text())
    rows = data["rows"]
    ks = [r["k"] for r in rows]
    oracle = [r["delta_oracle"] * 100 for r in rows]
    router = [r["delta_router"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ks, oracle, marker="o", label="oracle (top-K of true scores)")
    ax.plot(ks, router, marker="s", label="router (trained, 17k params)")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("top-K (log)")
    ax.set_ylabel("perplexity penalty (%)")
    ax.set_title(f"Layer {data['layer']} sweep, seq=512")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_sparsity_per_layer(
    sparsity_path: Path = Path("bench/results/sparsity.json"),
    out: Path = Path("bench/figures/sparsity_per_layer.png"),
) -> None:
    data = json.loads(sparsity_path.read_text())
    layers = data["layers"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k in [16, 32, 64, 128]:
        ys = [layer["avg_mass_top_k"][str(k)] for layer in layers]
        xs = [layer["layer"] for layer in layers]
        ax.plot(xs, ys, marker="o", label=f"top-{k}")
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("attention mass in top-K keys")
    ax.set_title(f"Sparsity per layer — Qwen-1.5B, seq={data['seq_len']}")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Generate every figure that has source data.")
    args = parser.parse_args()
    if args.all or True:
        plot_oracle_ceiling()
        plot_router_vs_oracle()
        plot_sparsity_per_layer()


if __name__ == "__main__":
    main()

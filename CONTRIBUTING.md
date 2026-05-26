# Contributing to OSSA

Thanks for your interest. OSSA is a research preview at v0.1.0; we welcome
both PRs and issues.

## Quick start for contributors

```bash
git clone https://github.com/narelabs/ossa
cd ossa
pip install -e ".[dev]"
pytest tests -q          # 7 tests should pass in ~3 s
```

## Where help is most useful

The current bottleneck is going from "research preview" to "fast library".
Roughly in priority order:

1. **Validate the Triton kernel** in `src/ossa/triton_kernel.py` on a
   real CUDA box. We wrote it on a Windows machine where Triton isn't
   available; a fallback runs the PyTorch implementation. The kernel
   compiles in our heads but has not actually been launched.
2. **Train all 28 routers.** `bench/multi_layer_train.py` already does
   this on a list of layers. Running on every layer for ~1500 steps
   gives the proper full-stack perplexity number we don't yet have.
3. **WikiText proper.** `bench/wikitext_eval.py` downloads the wikitext-2
   raw test split. Wire that into the headline numbers.
4. **Long context.** Re-train the router at `seq_len=2048` and 4096; the
   only thing tying us to 512 right now is the router's `pos_bias`
   embedding size.
5. **Other base models.** `Qwen2.5-1.5B-Instruct` is the headline LM.
   Drop-in support for Llama-3.2-3B and Mistral-7B is a half-day job.

## Style

- One module = one responsibility. `bench/foo.py` runs a benchmark and
  writes JSON; `src/ossa/foo.py` provides reusable building blocks.
- Tests are colocated under `tests/test_<module>.py` and must run on CPU
  in seconds — avoid loading the LM in unit tests.
- We don't require type hints everywhere but please add them on public
  APIs (the things imported from `ossa.__init__`).

## Reporting numbers

Keep the JSON output convention: every benchmark writes a structured
result to `bench/results/*.json`. README claims should be reproducible
from these files.

## License

By contributing you agree that your contribution is licensed under
Apache-2.0.

"""OSSA — Open Sparse Subquadratic Attention."""

from ossa.capture import capture_attention_scores, load_model
from ossa.sparse_attention import (
    dense_attention_reference,
    sparse_attention_forward,
    sparse_attention_forward_chunked,
)
from ossa.triton_kernel import HAS_TRITON, topk_attention

__all__ = [
    "capture_attention_scores",
    "load_model",
    "dense_attention_reference",
    "sparse_attention_forward",
    "sparse_attention_forward_chunked",
    "topk_attention",
    "HAS_TRITON",
]
__version__ = "0.1.0"

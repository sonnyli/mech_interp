from typing import Tuple

import einops
import torch as t
from jaxtyping import Float
from torch import Tensor
from transformer_lens import (
    FactoredMatrix,
    HookedTransformer,
)
from utils import norm


def get_effective_embedding(
    model: HookedTransformer,
) -> Float[Tensor, "d_vocab d_model"]:
    return model.blocks[0](model.W_E[:, None, :])[:, 0]


def get_full_ov_circuit_for_head(
    model: HookedTransformer,
    layer: int,
    head: int,
) -> FactoredMatrix:
    W_O, W_V = model.W_O[layer, head], model.W_V[layer, head]
    W_EE = get_effective_embedding(model)

    full_OV_circuit = FactoredMatrix(
        model.ln_final(W_EE @ W_V @ W_O), model.W_U
    )

    # think of this as a bi-linear function:
    # H_OV(t, u) = \
    #       L10H7's change in model output logits for token u (not accounting for layer norm) if
    #       L10H7 fully attends to source token t
    return full_OV_circuit


def ov_head_impact(
    circuit: FactoredMatrix, batch_size=2048
) -> Tuple[float, float]:
    d_vocab = circuit.shape[0]
    p05_k = int(d_vocab * 0.05)

    n_bottom_10 = 0
    n_bottom_p05 = 0
    idxs = t.arange(d_vocab, device=circuit.A.device)[:, None]
    for offset in range(0, d_vocab, batch_size):
        idx_slice = slice(offset, offset + batch_size)
        bottom_k_idx = (
            circuit[idx_slice].AB.topk(k=p05_k, largest=False).indices
        )
        in_bottom = idxs[idx_slice] == bottom_k_idx
        n_bottom_p05 += in_bottom.sum()
        n_bottom_10 += in_bottom[:, :10].sum()

    return n_bottom_10.item() / d_vocab, n_bottom_p05.item() / d_vocab


def get_full_qk_circuit_for_head(
    model: HookedTransformer,
    layer: int,
    head: int,
    query_use_effective_embedding: bool = False,
) -> FactoredMatrix:
    W_EE = model.blocks[0](model.W_E[:, None, :])[:, 0]

    qk_norm = model.cfg.d_model**0.5
    W_EE_norm = norm(W_EE) * qk_norm
    W_U_norm = norm(model.W_U.T) * qk_norm

    W_Q, W_K = model.W_Q[layer, head], model.W_K[layer, head]
    W_QK = FactoredMatrix(W_Q, W_K.T)

    if query_use_effective_embedding:
        qk_circuit = W_EE_norm @ W_QK @ W_EE_norm.T
    else:
        qk_circuit = W_U_norm @ W_QK @ W_EE_norm.T

    # think of this as a bi-linear function:
    # H_QK(d, s) = \
    #       the attention score for token s for L10H7, given destination token d
    return qk_circuit


def qk_circuit_self_attn(
    circuit: FactoredMatrix, batch_size=2048, k=20
) -> Float[Tensor, "k"]:
    """For an attn head, determine how much attention a token will pay to itself relative to other tokens."""
    d_vocab = circuit.shape[0]

    self_attn_rank = t.zeros((k,), dtype=t.long, device=circuit.A.device)
    idxs = t.arange(d_vocab, device=circuit.A.device)[:, None]
    for offset in range(0, d_vocab, batch_size):
        idx_slice = slice(offset, offset + batch_size)
        top_k_idx = circuit[idx_slice].AB.topk(k=k).indices
        in_top = idxs[idx_slice] == top_k_idx
        self_attn_rank += in_top.sum(dim=0)

    self_attn_rank[-1] = d_vocab - (self_attn_rank.sum() - self_attn_rank[-1])
    return self_attn_rank


def find_k_most_suppressed_by_ov(
    model: HookedTransformer,
    head_tup: Tuple[int, int],
    k_s: int,
    batch_size: int = 512,
) -> Float[Tensor, "d_vocab d_model k_s"]:
    """Retrieve the k_s most suppressed spans.

    For each token in the vocabulary, retrieve the unembeddings for the
    k_s most suppressed tokens as determined by the most negative logits in the OV circuit.
    """
    layer, head = head_tup

    ov_circuit = get_full_ov_circuit_for_head(model, layer, head)
    spans = t.empty(
        (model.cfg.d_vocab, model.cfg.d_model, k_s),
        dtype=t.float32,
        device=model.W_U.device,
    )

    for offset in range(0, model.cfg.d_vocab, batch_size):
        batch_idxs = slice(offset, offset + batch_size)
        neg_spans_idxs = (
            ov_circuit[batch_idxs].AB.topk(k=k_s, dim=-1, largest=False).indices
        )
        spans[batch_idxs] = einops.rearrange(
            model.W_U[:, neg_spans_idxs],
            "d_model batch k_suppressed -> batch d_model k_suppressed",
        )

    return spans


import circuitsvis as cv
import einops
import humanize
import torch as t
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def norm(tens: Float[Tensor, "... vec"]) -> Float[Tensor, "... vec"]:
    return tens / tens.norm(p=2, dim=-1, keepdim=True)


def update_running_mean(
    old_mean: float, n_old_examples: int, new_mean: float, n_new_examples: int
) -> float:
    return (old_mean * n_old_examples + new_mean * n_new_examples) / (
        n_old_examples + n_new_examples
    )


def tensor_size(tens: Tensor) -> str:
    return humanize.naturalsize(tens.numel() * tens.element_size())


def colored_tokens(tokens, values):
    return cv.tokens.colored_tokens(
        tokens,
        values,
        positive_color="#63A7CE",
        negative_color="#E7876B",
    )


def project_vectors_onto_spans(
    vals: Float[Tensor, "... d_model"],
    spans: Float[Tensor, "... d_model k"],
) -> Float[Tensor, "... d_model"]:
    sol = t.linalg.lstsq(spans, vals).solution
    assert (
        not sol.isnan().any()
    ), "NaN solution, likely rank deficient. Should not happen!"
    projected = einops.einsum(
        spans,
        sol,
        "... d_model k, ... k -> ... d_model",
    )
    return projected


def kl_div(
    clean_logits: Float[Tensor, "... logit"],
    ablated_logits: Float[Tensor, "... logit"],
) -> Float[Tensor, ""]:
    vocab_size = clean_logits.shape[-1]
    shape = (-1, vocab_size)

    clean_logprobs = clean_logits.reshape(shape).log_softmax(dim=-1)
    ablated_logprobs = ablated_logits.reshape(shape).log_softmax(dim=-1)

    return F.kl_div(
        ablated_logprobs,
        clean_logprobs,
        reduction="batchmean",
        log_target=True,
    )

from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple

import einops
import torch as t
import transformer_lens.utils as tlutils
from dataset import OpenWebTextDataset
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from transformer_lens import (
    HookedTransformer,
)
from transformer_lens.hook_points import HookPoint
from utils import (
    kl_div,
    norm,
    project_vectors_onto_spans,
    update_running_mean,
)

Hook = Callable[[Tensor, HookPoint], Tensor]


class Ablation:
    @abstractmethod
    def generate_fwd_hooks(self) -> list[Hook]:
        raise NotImplementedError


AblationConstructor = Callable[[Int[Tensor, "batch seq"]], Ablation]


def compare_ablations(
    model: HookedTransformer,
    ablations: dict[str, AblationConstructor],
    dataset: Dataset,
    n_examples: int,
    batch_size: int = 64,
    device="cuda",
) -> Tuple[dict[str, float], dict[str, float]]:
    """Calculates the losses and KL divergences from the clean model token distribution when running ablations.

    Returns:
    tuple:
        - losses (dict[str, float]): losses run on clean model + ablated models
        - kl_divs (dict[str, float]): KL divergences from clean model token distribution for ablated models
    """
    t.cuda.empty_cache()
    running_losses = defaultdict(float)
    running_kl_divs = defaultdict(float)
    n_examples = min(n_examples, len(dataset))

    dl = iter(DataLoader(dataset, batch_size=batch_size))
    total_examples = 0
    for _ in (pbar := tqdm(range(0, n_examples, batch_size))):
        tokens = next(dl).to(device)
        batch_examples = len(tokens)

        clean_logits, loss_clean = model(tokens, return_type="both")
        running_losses["clean"] = update_running_mean(
            running_losses["clean"], total_examples, loss_clean, batch_examples
        )

        for name, ablation_constructor in ablations.items():
            ablation_hooks = ablation_constructor(tokens).generate_fwd_hooks()
            ablated_logits, loss_ablated = model.run_with_hooks(
                tokens, fwd_hooks=ablation_hooks, return_type="both"
            )
            ablated_kl_div = kl_div(clean_logits, ablated_logits)
            running_losses[name] = update_running_mean(
                running_losses[name],
                total_examples,
                loss_ablated,
                batch_examples,
            )
            running_kl_divs[name] = update_running_mean(
                running_kl_divs[name],
                total_examples,
                ablated_kl_div,
                batch_examples,
            )

        # could use wandb, but this is a quick in-notebook way to monitor progress
        descriptions = [
            f"[{name}] loss: {val:.4f}" for name, val in running_losses.items()
        ] + [
            f"[{name}] KL div: {val:.4f}"
            for name, val in running_kl_divs.items()
        ]
        pbar.set_description("\t".join(descriptions))

        total_examples += batch_examples

    t.cuda.empty_cache()
    losses = {name: val.item() for name, val in running_losses.items()}
    kl_divs = {name: val.item() for name, val in running_kl_divs.items()}
    return losses, kl_divs


def hook_ablate_head_output(
    z: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    head: int,
    ablation_val: Float[Tensor, "d_head"],
):
    """Replaces the head output with the given ablation values."""
    z[:, :, head] = ablation_val
    return z


def hook_update_resid_stream_head_result(
    resid: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    W_O: Float[Tensor, "d_head d_model"],
    remove_z: Float[Tensor, "batch seq d_head"],
    add_z: Float[Tensor, "batch seq d_head"],
):
    """Adds and removes head outputs from the residual stream."""
    resid += (add_z - remove_z) @ W_O
    return resid


class CSPAblation(Ablation):
    def __init__(
        self,
        model: HookedTransformer,
        head_tup: Tuple[int, int],
        seq_toks: Int[Tensor, "batch seq"],
        k_suppressed: Float[Tensor, "d_vocab d_model k_suppressed"],
        head_mean_components: dict[str, Float[Tensor, "d_model"]],
        k_u: int = 10,
        apply_qk_ablation: bool = True,
        apply_ov_ablation: bool = True,
        ov_ablation_span_proj: bool = False,
    ) -> None:
        """Implements the CSP Ablation detailed in section 3.3.1."""
        layer, head = head_tup
        self.model = model
        self.W_U = model.W_U
        self.W_O = model.W_O[layer, head]
        self.n_layers = model.cfg.n_layers

        self.layer, self.head = layer, head
        self.seq_toks = seq_toks

        self.k_suppressed = k_suppressed
        self.k_u = k_u
        self.top_predicted_k = int(model.cfg.d_vocab * 0.05)
        self.head_mean_components = head_mean_components

        self.batch_size = self.seq_toks.shape[0]
        self.seq_len = self.seq_toks.shape[1]
        self.casual_attn_mask = t.ones(
            (self.seq_len, self.seq_len),
            dtype=t.bool,
            device=self.seq_toks.device,
        ).triu_(1)

        self.apply_qk_ablation = apply_qk_ablation
        self.apply_ov_ablation = apply_ov_ablation
        self.ov_ablation_span_proj = ov_ablation_span_proj

        # calculated when running the ablation
        self.head_components = {}
        self.head_input = None
        self.attn_pattern = None

    def generate_fwd_hooks(
        self,
    ) -> list[Tuple[str, Hook]]:
        """Creates all the hooks to be added to a HookedTransformer to run the ablation."""
        return [
            (
                tlutils.get_act_name("normalized", self.layer, "ln1"),
                self.hook_cache_head_input,
            ),
            (
                tlutils.get_act_name("pattern", self.layer),
                self.hook_cache_attn_pattern,
            ),
            (
                tlutils.get_act_name("resid_post", self.n_layers - 1),
                self.hook_inject_ablated_head_value,
            ),
        ] + [
            (
                tlutils.get_act_name(comp, self.layer),
                partial(self.hook_cache_head_component, component=comp),
            )
            for comp in ["q", "v", "z"]
        ]

    def hook_inject_ablated_head_value(
        self, resid: Float[Tensor, "batch seq d_model"], hook: HookPoint
    ) -> Float[Tensor, "batch seq d_model"]:
        """Top level hook for applying CSP ablation."""
        unweighted_z = self.head_components["v"] @ self.W_O
        if self.apply_ov_ablation:
            if self.ov_ablation_span_proj:
                unweighted_z = self.calculate_ov_ablated_head_results_span_proj(
                    unweighted_z
                )
            else:
                unweighted_z = self.calculate_ov_ablated_head_results(
                    unweighted_z
                )

        if self.apply_qk_ablation:
            ablated_head_results = self.calculate_qk_ablated_head_results(
                unweighted_z
            )
        else:
            ablated_head_results = einops.einsum(
                self.attn_pattern,
                unweighted_z,
                "batch seqQ seqK, batch seqK d_model -> batch seqQ d_model",
            )

        non_ablated_head_results = self.head_components["z"] @ self.W_O
        resid += ablated_head_results - non_ablated_head_results
        return resid

    def calculate_qk_ablated_head_results(
        self, unweighted_z: Float[Tensor, "batch seq d_model"]
    ) -> Float[Tensor, "batch seq d_model"]:
        """Applies QK ablation specified in the paper.

        This mean ablates the result vector for each destination-source pair, except for
        those that the destination token predicts the most.
        """
        mean_head_result = self.head_mean_components["v"] @ self.W_O

        mean_ablation_mask = self.get_qk_mean_ablation_mask()
        per_seq_mean_ablation_z = (self.attn_pattern * mean_ablation_mask).sum(
            dim=-1, keepdim=True
        ) * mean_head_result

        non_ablated_attn_pattern = t.where(
            mean_ablation_mask, 0.0, self.attn_pattern
        )
        non_ablated_z = einops.einsum(
            non_ablated_attn_pattern,
            unweighted_z,
            "batch seqQ seqK, batch seqK d_model -> batch seqQ d_model",
        )

        return non_ablated_z + per_seq_mean_ablation_z

    def get_qk_mean_ablation_mask(self) -> Bool[Tensor, "batch seqQ seqK"]:
        """Calculates the mask of dest->src pairs to mean ablate v-vectors for.

        These are the source tokens that the destination token most highly predicts
        based on the pre head destination logit lens.
        """

        # TODO(sonny): should we be applying the ln scale calculated from non ablated residual stream?
        top_dest_pred_tokens = einops.rearrange(
            (self.model.ln_final(self.head_input) @ self.W_U)
            .topk(k=self.top_predicted_k, dim=-1)
            .indices,
            "batch seqQ top_k -> batch seqQ 1 top_k",
        )
        src_tokens = einops.rearrange(
            self.seq_toks, "batch seqK -> batch 1 seqK 1"
        )
        mean_ablation_mask = (top_dest_pred_tokens == src_tokens).sum(
            dim=-1
        ) == 0
        attn_mask = einops.repeat(
            self.casual_attn_mask,
            "seqQ seqK -> batch seqQ seqK",
            batch=self.seq_toks.shape[0],
        )
        mean_ablation_mask[attn_mask] = True
        return mean_ablation_mask

    def calculate_ov_ablated_head_results(
        self,
        unweighted_z: Float[Tensor, "batch seq d_model"],
    ) -> Float[Tensor, "batch seq d_model"]:
        """Ablates the OV output of the head with the method detailed in the paper.

        We project each source token OV output to its (single) unembedding vector.
        """
        seq_unembeds = norm(
            einops.rearrange(
                self.W_U[:, self.seq_toks],
                "d_model batch seq -> batch seq d_model",
            )
        )
        dots = einops.einsum(
            unweighted_z,
            seq_unembeds,
            "batch seq d_model, batch seq d_model -> batch seq",
        )[:, :, None]
        z_proj = seq_unembeds * t.where(dots < 0, dots, 0.0)

        return z_proj

    def calculate_ov_ablated_head_results_span_proj(
        self,
        unweighted_z: Float[Tensor, "batch seq d_model"],
    ) -> Float[Tensor, "batch seq d_model"]:
        """Ablates the OV output of the head with the method detailed in the paper's website.

        This projects the OV output for each source token S to the span of the top k_s copy-suppressed
        tokens of S.
        """
        mean_z = self.get_mean_head_result(unweighted_z)

        # subtract mean before projecting onto k-suppression span
        unweighted_z = unweighted_z - mean_z

        # project head results to each src_k_suppressed
        src_k_suppressed_spans = self.k_suppressed[
            self.seq_toks
        ]  # (batch seq d_model k)
        unweighted_z = project_vectors_onto_spans(
            unweighted_z, src_k_suppressed_spans
        )

        # add back the mean
        unweighted_z += mean_z

        return unweighted_z

    def get_mean_head_result(
        self, unweighted_z: Float[Tensor, "batch seq d_model"]
    ) -> Float[Tensor, "d_model"]:
        n_els = self.batch_size * self.seq_len * (self.seq_len + 1) / 2
        return unweighted_z.sum(dim=(0, 1)) / n_els

    def hook_cache_head_input(
        self, resid: Float[Tensor, "batch seq d_model"], hook: HookPoint
    ):
        self.head_input = resid.clone()
        return resid

    def hook_cache_attn_pattern(
        self,
        pattern: Float[Tensor, "batch head seqQ seqK"],
        hook: HookPoint,
    ):
        self.attn_pattern = pattern[:, self.head].clone()

    def hook_cache_head_component(
        self,
        vals: Float[Tensor, "batch seq head d_head"],
        hook: HookPoint,
        component: str,
    ):
        self.head_components[component] = vals[:, :, self.head].clone()
        return vals


class HeadResultMeanAblation(Ablation):
    def __init__(
        self, head_tup: Tuple[int, int], mean: Float[Tensor, "d_head"]
    ) -> None:
        self.layer, self.head = head_tup
        self.mean = mean

    def generate_fwd_hooks(self) -> list[Callable[[Tensor, HookPoint], Tensor]]:
        return [
            (
                tlutils.get_act_name("z", self.layer),
                partial(
                    hook_ablate_head_output,
                    head=self.head,
                    ablation_val=self.mean,
                ),
            )
        ]


class HeadResultDirectPathMeanAblation(Ablation):
    def __init__(
        self,
        head_tup: Tuple[int, int],
        mean: Float[Tensor, "d_head"],
        model: HookedTransformer,
    ) -> None:
        self.layer, self.head = head_tup
        self.mean = mean
        self.W_O = model.W_O[self.layer, self.head]
        self.n_layers = model.cfg.n_layers
        self.z = None

    def generate_fwd_hooks(self) -> list[Callable[[Tensor, HookPoint], Tensor]]:
        return [
            (tlutils.get_act_name("z", self.layer), self.hook_cache_z),
            (
                tlutils.get_act_name("resid_post", self.n_layers - 1),
                self.hook_update_resid_stream,
            ),
        ]

    def hook_cache_z(
        self, z: Float[Tensor, "batch seq head d_head"], hook: HookPoint
    ):
        self.z = z[:, :, self.head]
        return z

    def hook_update_resid_stream(
        self,
        resid: Float[Tensor, "batch seq d_model"],
        hook: HookPoint,
    ):
        resid += (self.mean - self.z) @ self.W_O
        return resid


class HeadResultIndirectPathMeanAblation(Ablation):
    def __init__(
        self,
        head_tup: Tuple[int, int],
        mean: Float[Tensor, "d_head"],
        model: HookedTransformer,
    ) -> None:
        self.layer, self.head = head_tup
        self.mean = mean
        self.W_O = model.W_O[self.layer, self.head]
        self.n_layers = model.cfg.n_layers
        self.z = None

    def generate_fwd_hooks(self) -> list[Callable[[Tensor, HookPoint], Tensor]]:
        return [
            (tlutils.get_act_name("z", self.layer), self.hook_ablate_head),
            (
                tlutils.get_act_name("resid_post", self.n_layers - 1),
                self.hook_update_resid_stream,
            ),
        ]

    def hook_ablate_head(
        self, z: Float[Tensor, "batch seq head d_head"], hook: HookPoint
    ):
        self.z = z[:, :, self.head].clone()
        z[:, :, self.head] = self.mean
        return z

    def hook_update_resid_stream(
        self,
        resid: Float[Tensor, "batch seq d_model"],
        hook: HookPoint,
    ):
        resid += (self.z - self.mean) @ self.W_O
        return resid


def get_mean_head_components(
    model: HookedTransformer,
    dataset: OpenWebTextDataset,
    head_tup: Tuple[int, int],
    num_examples: int,
    batch_size=64,
    device="cuda",
    components: Optional[list[str]] = None,
) -> dict[str, Float[Tensor, "d_head"]]:
    """Get the average value of head components run over a number of examples."""
    layer, head = head_tup
    if components is None:
        components = ["q", "k", "v", "z"]
    component_names = {tlutils.get_act_name(comp, layer) for comp in components}

    def cache_filter(name) -> bool:
        return name in component_names

    running_means = {
        component: t.zeros((model.cfg.d_head,), dtype=t.float32, device=device)
        for component in components
    }
    total_outputs = 0

    num_examples = min(num_examples, len(dataset))
    owt = iter(DataLoader(dataset, batch_size))
    for _ in tqdm(range(0, num_examples, batch_size)):
        tokens = next(owt)
        batch_outputs = tokens.numel()
        _, cache = model.run_with_cache(
            tokens,
            return_type="None",
            names_filter=cache_filter,
        )
        for comp in components:
            output = cache[comp, layer][:, :, head]
            running_means[comp][:] = (
                running_means[comp] * total_outputs + output.sum((0, 1))
            ) / (total_outputs + batch_outputs)

        total_outputs += batch_outputs

    return running_means


def get_model_dataset_per_token_losses(
    model: HookedTransformer,
    dataset: Dataset,
    ablation: AblationConstructor,
    num_examples: int,
    seq_len: int,
    batch_size=64,
    loss_device="cuda",
) -> Tuple[Float[Tensor, "batch seq"], Float[Tensor, "batch seq"]]:
    """Get the per token losses over a dataset on the clean and ablated model.

    Returns:
    tuple:
        - Tuple[Float[Tensor, "batch seq-1"]: clean model per token losses
        - Tuple[Float[Tensor, "batch seq-1"]: ablated model per token losses
    """
    clean_losses = t.empty(
        (num_examples, seq_len - 1), dtype=t.float32, device=loss_device
    )
    ablated_losses = t.empty(
        (num_examples, seq_len - 1), dtype=t.float32, device=loss_device
    )

    num_examples = min(num_examples, len(dataset))
    owt = iter(DataLoader(dataset, batch_size))
    for offset in tqdm(range(0, num_examples, batch_size)):
        tokens = next(owt)
        idxs = slice(offset, offset + len(tokens))
        clean_losses[idxs] = model(
            tokens,
            return_type="loss",
            loss_per_token=True,
        )

        ablated_losses[idxs] = model.run_with_hooks(
            tokens,
            return_type="loss",
            loss_per_token=True,
            fwd_hooks=ablation(tokens).generate_fwd_hooks(),
        )

    return clean_losses, ablated_losses


def top_completion_deltas_after_ablation(
    loss_delta: Float[Tensor, "batch seq"],
    k: int = 1024,
) -> Tuple[Float[Tensor, "k"], Tuple[Int[Tensor, "k"], Int[Tensor, "k"]]]:
    topk = loss_delta.flatten().topk(k=k)
    indices = t.unravel_index(topk.indices, loss_delta.shape)

    return topk.values, indices


def find_copy_suppression_examples_in_top_completions(
    model: HookedTransformer,
    head_tup: Tuple[int, int],
    dataset: OpenWebTextDataset,
    top_completion_idxs: Tuple[Int[Tensor, "example"], Int[Tensor, "example"]],
    batch_size=64,
    device="cuda",
) -> Tuple[
    Bool[Tensor, "completion"],
    Int[Tensor, "completion k"],
    Float[Tensor, "completion seq seq"],
    Int[Tensor, "completion k"],
]:
    """Given the top completions, determine which are deemed as copy suppression.

    Returns:
    tuple:
        - Bool[Tensor, "completion"]: bool tensor that indicates whether a completion is copy suppression
        - Int[Tensor, "completion k"]: tensor of top predictions pre-head at the destination token
        - Float[Tensor, "completion seq seq"]: tensor of attention patterns for the head
        - Int[Tensor, "completion k"]: tensor of top tokens suppressed by head
    """
    layer, head = head_tup
    cache_set = set(
        [
            tlutils.get_act_name("z", layer),
            tlutils.get_act_name("resid_pre", layer),
            tlutils.get_act_name("pattern", layer),
            tlutils.get_act_name("scale"),
        ]
    )

    prompt_idxs, tok_idxs = top_completion_idxs
    n_top = len(prompt_idxs)
    is_copy_suppression = t.zeros((n_top,), dtype=t.bool, device=device)
    k_top_logits = 10
    all_top_prehead_preds = t.zeros(
        (n_top, k_top_logits), dtype=t.long, device=device
    )
    all_head_top_negative_preds = t.zeros(
        (n_top, k_top_logits), dtype=t.long, device=device
    )
    attn_patterns = t.zeros(
        (n_top, dataset.max_seq_len, dataset.max_seq_len),
        dtype=t.float32,
        device=device,
    )
    for offset in tqdm(range(0, n_top, batch_size)):
        n_prompts = min(batch_size, n_top - offset)
        prompt_completion_idxs = tok_idxs[offset : offset + n_prompts]
        completion_idx_tup = (range(n_prompts), prompt_completion_idxs)
        tokens = dataset[prompt_idxs[offset : offset + n_prompts]].to(device)
        _, cache = model.run_with_cache(
            tokens,
            return_type=None,
            names_filter=lambda name: name in cache_set,
        )

        # 2.1.1
        # using the input residual stream to the head, determine if the highest predicted logits include
        # tokens that appeared in the context
        pre_head_logits = (
            cache.apply_ln_to_stack(cache["resid_pre", layer], layer=-1)[
                completion_idx_tup
            ]
            @ model.W_U
        )
        top_prehead_preds = pre_head_logits.topk(k=k_top_logits, dim=-1).indices
        all_top_prehead_preds[offset : offset + n_prompts] = top_prehead_preds
        top_in_context = (
            tokens[:, :, None] == top_prehead_preds[:, None, :]
        ).sum(dim=-1)
        has_top_in_context = (
            top_in_context.cumsum(dim=-1)[completion_idx_tup] > 0
        )

        # 2.1.2
        # determine if any of the source tokens were in the top 2 most attended to by the head
        pattern = cache["pattern", layer][:, head]
        attn_patterns[offset : offset + n_prompts] = pattern
        top_attn_to = (
            pattern[t.arange(n_prompts), prompt_completion_idxs]
            .topk(k=2)
            .indices
        )
        source_is_attended_to_most = (
            top_in_context.gather(-1, top_attn_to).sum(dim=-1) > 0
        )

        # 2.1.3
        # determine if the source token logits were in the top most negatively impacted
        # by the head
        head_out = einops.einsum(
            cache["z", layer][:, :, head],
            model.W_O[layer, head],
            "batch seq d_head, d_head d_embed -> batch seq d_embed",
        )
        head_logits = (
            cache.apply_ln_to_stack(head_out, layer=-1)[
                range(n_prompts), prompt_completion_idxs
            ]
            @ model.W_U
        )

        top_head_logit_decr = head_logits.topk(
            k=k_top_logits, largest=False
        ).indices
        all_head_top_negative_preds[offset : offset + n_prompts] = (
            top_head_logit_decr
        )
        head_bottom_in_context = (
            tokens[:, :, None] == top_head_logit_decr[:, None, :]
        ).sum(dim=-1)

        head_decr_source_token_logit = (
            head_bottom_in_context.cumsum(-1)[completion_idx_tup] > 0
        )

        is_copy_suppression[offset : offset + n_prompts] = (
            has_top_in_context
            & source_is_attended_to_most
            & head_decr_source_token_logit
        )

    return (
        is_copy_suppression,
        all_top_prehead_preds,
        attn_patterns,
        all_head_top_negative_preds,
    )

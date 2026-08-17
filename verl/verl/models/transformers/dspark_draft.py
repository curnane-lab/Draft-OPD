# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSpark draft model for the composed DFlash OPD student.

Vendored from DeepSeek's DeepSpec reference implementation
(https://github.com/deepseek-ai/DeepSpec, ``deepspec/modeling/dspark/``),
trimmed to what the on-policy distillation path needs:

- the draft backbone (target-context attention + Markov/confidence heads) with
  module names identical to the official checkpoints (e.g.
  ``deepseek-ai/dspark_qwen3_4b_block7``), so ``from_pretrained`` loads them
  directly — those checkpoints ship no remote code and plain ``AutoModel``
  would silently instantiate a stock Qwen3 backbone;
- a ``forward`` matching the composed student's call convention
  (``noise_embedding``/``target_hidden`` in, hidden states out) and a
  ``predict_confidence`` matching the student's head contract.

The offline-training pieces of the reference (anchor sampling, loss-mask
handling, eval helpers) are intentionally not vendored: the composed student
already has its own anchor/noise/mask machinery.
"""

from typing import Callable, Optional

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DSparkAttention(nn.Module):
    """Qwen3 attention whose K/V are the concatenation of the target-model
    context features and the draft (noise) tokens."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        layer_types = getattr(config, "layer_types", None) or []
        self.sliding_window = (
            config.sliding_window if layer_types and layer_types[layer_idx] == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden_states)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        if self.config._attn_implementation == "flex_attention" and self.num_key_value_groups > 1:
            kv_seq_len = k.shape[-2]
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
            k = k.reshape(bsz, self.num_attention_heads, kv_seq_len, self.head_dim)
            v = v.reshape(bsz, self.num_attention_heads, kv_seq_len, self.head_dim)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_is_causal = bool(kwargs.get("is_causal", False))
        # The SDPA path may consult module.is_causal when dispatching kernels,
        # so keep the per-call value mirrored on the module before invoking it.
        self.is_causal = attn_is_causal
        kwargs["is_causal"] = attn_is_causal
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.num_attention_heads * self.head_dim)
        return self.o_proj(attn_output), attn_weights


class DSparkDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DSparkAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DSparkVanillaMarkovHead(nn.Module):
    """Low-rank bigram logit bias: Embedding(vocab, rank) + Linear(rank -> vocab).

    Module names match the official DSpark checkpoints
    (``markov_head.markov_w1`` / ``markov_head.markov_w2`` under the draft).
    """

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        if self.markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {self.markov_rank}")
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_logits.size(2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states)


class DSparkAcceptRatePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features):
        return self.proj(features).squeeze(-1)


def build_dspark_markov_head(config) -> Optional[nn.Module]:
    markov_rank = int(getattr(config, "markov_rank", 0) or 0)
    if markov_rank == 0:
        return None
    markov_head_type = str(getattr(config, "markov_head_type", "vanilla") or "vanilla").lower()
    if markov_head_type != "vanilla":
        raise NotImplementedError(
            f"DSparkDraftModel supports only the 'vanilla' Markov head, got {markov_head_type!r}."
        )
    return DSparkVanillaMarkovHead(vocab_size=config.vocab_size, markov_rank=markov_rank)


def draft_config_has_dspark_markers(config) -> bool:
    """Detect DSpark draft checkpoints that ship no remote code.

    Checkpoints with their own remote code (``auto_map``, e.g. DFlash drafts)
    are left to the regular ``AutoModel`` path.
    """
    auto_map = getattr(config, "auto_map", None)
    if isinstance(auto_map, dict) and auto_map.get("AutoModel"):
        return False
    architectures = [str(arch) for arch in (getattr(config, "architectures", None) or [])]
    if any("DSpark" in arch for arch in architectures):
        return True
    return bool(getattr(config, "markov_rank", 0) or 0) or bool(getattr(config, "enable_confidence_head", False))


class DSparkDraftModel(Qwen3PreTrainedModel):
    """DSpark block-draft backbone (DeepSeek DeepSpec reference architecture).

    Loads the official checkpoints directly: module tree matches
    ``embed_tokens`` / ``layers`` / ``norm`` / ``fc`` / ``hidden_norm`` /
    ``lm_head`` / ``markov_head`` / ``confidence_head``.
    """

    _no_split_modules = ["DSparkDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        for field in ("target_layer_ids", "block_size"):
            if getattr(config, field, None) is None:
                raise ValueError(f"DSpark draft config.{field} must be provided.")
        self.target_layer_ids = [int(layer_id) for layer_id in config.target_layer_ids]

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [DSparkDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = getattr(config, "mask_token_id", None)

        self.markov_head = build_dspark_markov_head(config)

        self.enable_confidence_head = bool(getattr(config, "enable_confidence_head", False))
        self.confidence_head_with_markov = bool(getattr(config, "confidence_head_with_markov", False))
        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError("confidence_head_with_markov=True requires markov_rank > 0.")
                input_dim += int(config.markov_rank)
            self.confidence_head = DSparkAcceptRatePredictor(input_dim=input_dim)
        self.post_init()

    def forward(
        self,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Composed-student call convention: noise/target embeddings in, hidden out."""
        if noise_embedding is None or target_hidden is None:
            raise ValueError("DSparkDraftModel.forward requires noise_embedding and target_hidden.")
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        features = hidden_states
        if self.confidence_head_with_markov:
            if prev_token_ids is None or self.markov_head is None:
                raise ValueError("confidence_head_with_markov=True requires prev_token_ids and a Markov head.")
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(dtype=hidden_states.dtype)
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return self.confidence_head(features)

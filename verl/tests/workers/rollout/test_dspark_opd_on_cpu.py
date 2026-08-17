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
"""CPU tests for DSpark draft support in the composed DFlash OPD student."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from transformers import PretrainedConfig, Qwen3Config

from verl.models.transformers.dflash_student import (
    ComposedDFlashStudentForCausalLM,
    StudentVanillaMarkovHead,
    resolve_target_layer_ids,
)
from verl.models.transformers.dspark_draft import DSparkDraftModel, draft_config_has_dspark_markers
from verl.trainer.distillation.losses import distillation_loss, get_dspark_confidence_stream
from verl.trainer.ppo.core_algos import kl_penalty


def _bare_student(draft_model=None, config=None):
    student = object.__new__(ComposedDFlashStudentForCausalLM)
    student.config = config if config is not None else PretrainedConfig()
    student.draft_model = draft_model if draft_model is not None else SimpleNamespace(config=SimpleNamespace())
    return student


def _plain_draft(dflash_config=None, **attrs):
    config = SimpleNamespace(dflash_config=dflash_config or {})
    return SimpleNamespace(config=config, **attrs)


def _flat_dspark_draft(**config_attrs):
    """Mimic DSpark checkpoints (e.g. deepseek-ai/dspark_qwen3_4b_block7): no
    nested dflash_config; the DFlash-style keys live at the config top level."""
    return SimpleNamespace(config=SimpleNamespace(dflash_config=None, **config_attrs))


def _make_markov_head(vocab_size=8, markov_rank=2):
    head = StudentVanillaMarkovHead(vocab_size=vocab_size, markov_rank=markov_rank)
    with torch.no_grad():
        head.markov_w1.weight.copy_(
            torch.arange(vocab_size * markov_rank, dtype=torch.float32).reshape(vocab_size, markov_rank) / 10.0
        )
        head.markov_w2.weight.copy_(
            torch.arange(vocab_size * markov_rank, dtype=torch.float32).reshape(vocab_size, markov_rank) / 20.0
            + 0.05
        )
    return head


def test_draft_variant_defaults_to_dflash_without_markers():
    student = _bare_student()
    assert student._get_draft_variant() == "dflash"
    assert not student._is_dspark_markov_enabled()
    assert not student._is_dspark_confidence_enabled()


def test_draft_variant_auto_detects_dspark_markers():
    by_markov_rank = _bare_student(draft_model=_plain_draft({"markov_rank": 256}))
    assert by_markov_rank._get_draft_variant() == "dspark"

    by_confidence_flag = _bare_student(draft_model=_plain_draft({"enable_confidence_head": True}))
    assert by_confidence_flag._get_draft_variant() == "dspark"

    by_projector = _bare_student(draft_model=_plain_draft({"projector_type": "dspark"}))
    assert by_projector._get_draft_variant() == "dspark"

    by_head_attr = _bare_student(draft_model=_plain_draft({}, markov_head=_make_markov_head()))
    assert by_head_attr._get_draft_variant() == "dspark"


def test_draft_variant_config_and_env_override(monkeypatch):
    config = PretrainedConfig()
    config.verl_dflash_draft_variant = "dflash"
    student = _bare_student(draft_model=_plain_draft({"markov_rank": 256}), config=config)
    assert student._get_draft_variant() == "dflash"

    config.verl_dflash_draft_variant = "dspark"
    student = _bare_student(config=config)
    assert student._get_draft_variant() == "dspark"

    config.verl_dflash_draft_variant = "not-a-variant"
    student = _bare_student(draft_model=_plain_draft({"markov_rank": 256}), config=config)
    assert student._get_draft_variant() == "dflash"

    monkeypatch.setenv("VERL_DFLASH_DRAFT_VARIANT", "dspark")
    student = _bare_student()
    assert student._get_draft_variant() == "dspark"


def test_dspark_flat_config_mask_token_id_view_and_variant():
    draft = _flat_dspark_draft(mask_token_id=151669, markov_rank=256, enable_confidence_head=True)
    student = _bare_student(draft_model=draft)
    assert student._get_mask_token_id() == 151669
    view = student._get_dspark_dflash_config()
    assert view["markov_rank"] == 256
    assert view["enable_confidence_head"] is True
    # flat markers also drive variant auto-detection
    assert student._get_draft_variant() == "dspark"


def test_mask_token_id_nested_dict_shape_unchanged():
    student = _bare_student(draft_model=_plain_draft({"mask_token_id": 151669}))
    assert student._get_mask_token_id() == 151669


def test_mask_token_id_missing_raises():
    student = _bare_student()
    with pytest.raises(ValueError, match="mask_token_id"):
        student._get_mask_token_id()


def test_resolve_target_layer_ids_flat_config():
    main_model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=36))
    draft = _flat_dspark_draft(target_layer_ids=[1, 9, 17, 25, 33], num_hidden_layers=5)
    assert resolve_target_layer_ids(main_model, draft) == [1, 9, 17, 25, 33]


def _tiny_dspark_draft_config():
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=128,
    )
    config.block_size = 3
    config.target_layer_ids = [1, 3]
    config.mask_token_id = 63
    config.num_anchors = 4
    config.markov_rank = 4
    config.markov_head_type = "vanilla"
    config.enable_confidence_head = True
    config.confidence_head_with_markov = True
    config._attn_implementation = "eager"
    return config


def test_draft_config_has_dspark_markers_detection():
    by_arch = PretrainedConfig()
    by_arch.architectures = ["Qwen3DSparkModel"]
    assert draft_config_has_dspark_markers(by_arch)

    # DFlash checkpoints ship their own remote code: stay on the AutoModel path.
    with_remote_code = PretrainedConfig()
    with_remote_code.architectures = ["DFlashDraftModel"]
    with_remote_code.auto_map = {"AutoModel": "dflash.DFlashDraftModel"}
    assert not draft_config_has_dspark_markers(with_remote_code)

    by_flat_keys = PretrainedConfig()
    by_flat_keys.markov_rank = 256
    assert draft_config_has_dspark_markers(by_flat_keys)

    assert not draft_config_has_dspark_markers(PretrainedConfig())


def test_dspark_draft_forward_matches_student_call_convention():
    torch.manual_seed(0)
    config = _tiny_dspark_draft_config()
    model = DSparkDraftModel(config).eval()
    batch, ctx_len, q_len = 2, 7, 6
    noise_embedding = torch.randn(batch, q_len, config.hidden_size)
    target_hidden = torch.randn(batch, ctx_len, len(config.target_layer_ids) * config.hidden_size)
    position_ids = torch.arange(ctx_len + q_len).unsqueeze(0).expand(batch, -1)
    with torch.no_grad():
        out = model(
            position_ids=position_ids,
            attention_mask=None,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            use_cache=False,
        )
    assert out.shape == (batch, q_len, config.hidden_size)


def test_dspark_draft_checkpoint_weight_name_contract():
    model = DSparkDraftModel(_tiny_dspark_draft_config())
    keys = set(model.state_dict())
    expected = {
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.1.self_attn.o_proj.weight",
    }
    assert expected <= keys
    # embed_tokens/lm_head are deliberately omitted: the composed student uses
    # the frozen target model's embedding and output head, and the rollout
    # engine loads its own copies from the checkpoint (DeepSpec keeps them
    # frozen). Not carrying them shrinks the FSDP flat-parameter peak.
    assert "embed_tokens.weight" not in keys
    assert "lm_head.weight" not in keys
    # Official checkpoints do contain those tensors; loading must tolerate them.
    extra = {"embed_tokens.weight": torch.zeros(64, 16), "lm_head.weight": torch.zeros(64, 16)}
    _, unexpected = model.load_state_dict({**model.state_dict(), **extra}, strict=False)
    assert set(unexpected) == set(extra)
    # confidence_head_with_markov widens the predictor input by the Markov rank
    config = _tiny_dspark_draft_config()
    assert model.confidence_head.proj.in_features == config.hidden_size + config.markov_rank


def test_dspark_draft_predict_confidence_with_markov():
    model = DSparkDraftModel(_tiny_dspark_draft_config()).eval()
    hidden = torch.randn(2, 5, 16)
    prev_token_ids = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        confidence = model.predict_confidence(hidden, prev_token_ids=prev_token_ids)
    assert confidence.shape == (2, 5)
    with pytest.raises(ValueError, match="prev_token_ids"):
        model.predict_confidence(hidden)


def test_dspark_draft_from_pretrained_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = DSparkDraftModel(_tiny_dspark_draft_config()).eval()
    model.save_pretrained(tmp_path)
    loaded = DSparkDraftModel.from_pretrained(tmp_path)
    loaded.config._attn_implementation = "eager"
    loaded.eval()
    noise_embedding = torch.randn(1, 3, 16)
    target_hidden = torch.randn(1, 5, 32)
    position_ids = torch.arange(8).unsqueeze(0)
    with torch.no_grad():
        ref = model(position_ids=position_ids, noise_embedding=noise_embedding, target_hidden=target_hidden)
        out = loaded(position_ids=position_ids, noise_embedding=noise_embedding, target_hidden=target_hidden)
    assert torch.allclose(ref, out, atol=1e-6)


def test_dspark_prev_token_chain_matches_specforge_contract():
    student = _bare_student()
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=torch.long)
    anchor_positions = torch.tensor([[2, 5]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True, True]])

    prev = student._create_prev_token_ids_for_anchors(
        input_ids=input_ids,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        block_size=4,
    )

    # Block layout here is [anchor, draft_1, ..., draft_{B-1}]; SpecForge blocks
    # drop the anchor slot, so SpecForge position k == draft position k + 1 and
    # both use input_ids[anchor + k] as the prev token (anchor token first,
    # then the accepted-prefix tokens from the recorded response).
    assert prev.tolist() == [[11, 12, 13, 14, 14, 15, 16, 17]]
    for block_idx, anchor in enumerate([2, 5]):
        for draft_offset in range(1, 4):
            assert prev[0, block_idx * 4 + draft_offset].item() == input_ids[0, anchor + draft_offset - 1].item()

    # Masked blocks are zeroed, and anchor 0 clamps instead of going negative.
    masked = student._create_prev_token_ids_for_anchors(
        input_ids=input_ids,
        anchor_positions=torch.tensor([[0, 5]], dtype=torch.long),
        block_keep_mask=torch.tensor([[True, False]]),
        block_size=4,
    )
    assert masked[0, 4:].tolist() == [0, 0, 0, 0]
    assert masked[0, 0].item() == 10


def test_dspark_markov_bias_prefers_draft_head_and_matches_vanilla_formula():
    draft_head = _make_markov_head()
    fallback_head = _make_markov_head()
    with torch.no_grad():
        fallback_head.markov_w2.weight.mul_(2.0)
    student = _bare_student(draft_model=_plain_draft({}, markov_head=draft_head))
    student.__dict__["dspark_markov_head"] = fallback_head

    logits = torch.randn(5, 8)
    hidden = torch.randn(5, 8)
    prev_token_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    biased = student._apply_dspark_markov_bias(logits, hidden, prev_token_ids)
    expected = logits + draft_head.markov_w2(draft_head.markov_w1(prev_token_ids))
    assert torch.allclose(biased, expected)


def test_dspark_markov_bias_uses_student_fallback_head():
    fallback_head = _make_markov_head()
    student = _bare_student()
    student.__dict__["dspark_markov_head"] = fallback_head

    logits = torch.randn(3, 8)
    hidden = torch.randn(3, 8)
    prev_token_ids = torch.tensor([5, 6, 7], dtype=torch.long)

    biased = student._apply_dspark_markov_bias(logits, hidden, prev_token_ids)
    expected = logits + fallback_head.compute_step_bias(prev_token_ids)
    assert torch.allclose(biased, expected)


def test_dspark_markov_bias_falls_back_to_apply_logits_head():
    def apply_logits_head(logits, *, prev_token_ids=None, hidden_states=None, **kwargs):
        return logits + 7.0

    student = _bare_student(draft_model=_plain_draft({}, apply_logits_head=apply_logits_head))
    logits = torch.randn(2, 8)
    biased = student._apply_dspark_markov_bias(logits, torch.randn(2, 8), torch.tensor([1, 2]))
    assert torch.allclose(biased, logits + 7.0)


def test_dspark_markov_bias_rejects_rnn_head():
    rnn_head = SimpleNamespace(
        markov_head_type="rnn",
        compute_step_bias=lambda token_ids, hidden_states: torch.zeros(2, 8),
    )
    student = _bare_student(draft_model=_plain_draft({}, markov_head=rnn_head))
    with pytest.raises(NotImplementedError):
        student._apply_dspark_markov_bias(torch.randn(2, 8), torch.randn(2, 8), torch.tensor([1, 2]))


def test_selected_lm_log_probs_include_markov_bias():
    head = _make_markov_head(vocab_size=11, markov_rank=2)
    student = _bare_student(draft_model=_plain_draft({}, markov_head=head))
    draft_hidden = torch.randn(2, 5, 8, dtype=torch.float32)
    output_embeddings = torch.nn.Linear(8, 11, bias=False)
    batch_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    draft_indices = torch.tensor([1, 3, 0, 4], dtype=torch.long)
    token_ids = torch.tensor([2, 5, 7, 1], dtype=torch.long)
    markov_prev = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=torch.long)

    # chunk_size=1 forces the multi-chunk path so the bias is applied per chunk.
    selected, entropy = student._compute_selected_lm_log_probs(
        draft_hidden=draft_hidden,
        output_embeddings=output_embeddings,
        batch_indices=batch_indices,
        draft_indices=draft_indices,
        token_ids=token_ids,
        chunk_size=1,
        calculate_entropy=True,
        markov_prev_token_ids=markov_prev,
    )

    full_logits = output_embeddings(draft_hidden) + head.compute_step_bias(markov_prev)
    full_log_probs = torch.log_softmax(full_logits.float(), dim=-1)
    expected = full_log_probs[batch_indices, draft_indices, token_ids]
    assert torch.allclose(selected, expected)
    selected_log_probs = full_log_probs[batch_indices, draft_indices]
    expected_entropy = -(selected_log_probs.exp() * selected_log_probs).sum(dim=-1)
    assert entropy is not None
    assert torch.allclose(entropy, expected_entropy)


def _collect_confidence(student, draft_hidden, *, offsets, anchor_positions=((3,)), block_size=5):
    batch_size = draft_hidden.shape[0]
    anchor_tensor = torch.tensor([list(anchor_positions)], dtype=torch.long)
    keep_tensor = torch.ones_like(anchor_tensor, dtype=torch.bool)
    width = len(offsets)
    return student._collect_dspark_confidence_outputs(
        draft_hidden=draft_hidden,
        prompt_lengths=torch.tensor([2] * batch_size, dtype=torch.long),
        response_lengths=torch.tensor([6] * batch_size, dtype=torch.long),
        anchor_positions=anchor_tensor,
        block_keep_mask=keep_tensor,
        draft_block_size=block_size,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=torch.tensor([[1] * width], dtype=torch.long),
        rejected_draft_offsets=torch.tensor([list(offsets)], dtype=torch.long),
        rejected_draft_mask=torch.tensor([[True] * width]),
    )


def _confidence_student(hidden_size=1):
    confidence_head = torch.nn.Linear(hidden_size, 1)
    with torch.no_grad():
        confidence_head.weight.fill_(1.0)
        confidence_head.bias.zero_()
    draft = _plain_draft({}, confidence_head=confidence_head, confidence_head_with_markov=False)
    return _bare_student(draft_model=draft), confidence_head


def test_dspark_confidence_labels_reject_at_last_draft_position():
    # offset = block_size - 1: every earlier draft position was accepted.
    student, confidence_head = _confidence_student()
    block_size = 5
    draft_hidden = torch.arange(block_size, dtype=torch.float32).view(1, block_size, 1) + 1.0

    logits, labels, mask = _collect_confidence(student, draft_hidden, offsets=[4])

    assert labels.tolist() == [[[0.0, 1.0, 1.0, 1.0, 0.0]]]
    assert mask.tolist() == [[[False, True, True, True, True]]]
    expected_logits = confidence_head(draft_hidden[0, 1:5]).squeeze(-1)
    assert torch.allclose(logits[0, 0, 1:5], expected_logits)


def test_dspark_confidence_labels_first_position_rejection():
    # offset = 1: only the first draft position is labeled, with 0.
    student, confidence_head = _confidence_student()
    block_size = 5
    draft_hidden = torch.arange(block_size, dtype=torch.float32).view(1, block_size, 1) + 1.0

    logits, labels, mask = _collect_confidence(student, draft_hidden, offsets=[1])

    assert labels.tolist() == [[[0.0, 0.0, 0.0, 0.0, 0.0]]]
    assert mask.tolist() == [[[False, True, False, False, False]]]
    assert torch.allclose(logits[0, 0, 1], confidence_head(draft_hidden[0, 1]).squeeze(-1))


def test_dspark_confidence_labels_mid_block_rejection():
    student, _ = _confidence_student()
    block_size = 6
    draft_hidden = torch.zeros(1, block_size, 1)

    _, labels, mask = _collect_confidence(student, draft_hidden, offsets=[3], block_size=block_size)

    assert labels.tolist() == [[[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]]]
    assert mask.tolist() == [[[False, True, True, True, False, False]]]


def test_dspark_confidence_empty_metadata_returns_empty_stream():
    student, _ = _confidence_student()
    draft_hidden = torch.zeros(1, 5, 1)
    logits, labels, mask = student._collect_dspark_confidence_outputs(
        draft_hidden=draft_hidden,
        prompt_lengths=torch.tensor([2]),
        response_lengths=torch.tensor([6]),
        anchor_positions=torch.tensor([[3]]),
        block_keep_mask=torch.tensor([[True]]),
        draft_block_size=5,
        max_tokens_per_sample=None,
        rejected_draft_anchor_indices=torch.tensor([[-2]]),
        rejected_draft_offsets=torch.tensor([[-1]]),
        rejected_draft_mask=torch.tensor([[False]]),
    )
    assert logits.shape == (1, 1, 5)
    assert labels.shape == (1, 1, 5)
    assert not bool(mask.any())
    assert get_dspark_confidence_stream(
        {
            "opd_dspark_confidence_logits": logits,
            "opd_dspark_confidence_labels": labels,
            "opd_dspark_confidence_mask": mask,
        }
    ) is None


def test_dspark_confidence_stream_validates_shapes():
    assert get_dspark_confidence_stream({}) is None
    with pytest.raises(ValueError):
        get_dspark_confidence_stream(
            {
                "opd_dspark_confidence_logits": torch.zeros(1, 1, 4),
                "opd_dspark_confidence_labels": torch.zeros(1, 1, 5),
                "opd_dspark_confidence_mask": torch.ones(1, 1, 4, dtype=torch.bool),
            }
        )
    with pytest.raises(RuntimeError):
        get_dspark_confidence_stream({"opd_dspark_confidence_mask": torch.ones(1, 1, 4, dtype=torch.bool)})


def _single_sequence_logprobs(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    return torch.nested.as_nested_tensor([values], layout=torch.jagged)


def _confidence_loss_inputs(confidence_loss_weight, confidence_tv_target_mix=0.0):
    data = TensorDict(
        {
            "prompts": torch.tensor([[1]], dtype=torch.long),
            "responses": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "teacher_logprobs": _single_sequence_logprobs([-0.5, -0.7, 0.0]),
        },
        batch_size=[1],
    )
    # Slots are indexed by in-block draft offset: slot 0 is the anchor position
    # and stays unlabeled; here offset=2, so slot 1 was accepted and slot 2 rejected.
    confidence_logits = torch.tensor([[[0.0, 0.9, -1.2, 0.0]]], dtype=torch.float32)
    confidence_labels = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    confidence_mask = torch.tensor([[[False, True, True, False]]])
    model_output = {
        "log_probs": torch.tensor([-0.4, -0.9, 0.0], dtype=torch.float32),
        "opd_loss_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "opd_dspark_confidence_logits": confidence_logits,
        "opd_dspark_confidence_labels": confidence_labels,
        "opd_dspark_confidence_mask": confidence_mask,
        "opd_dspark_confidence_token_count": torch.tensor(2.0),
    }
    config = SimpleNamespace(loss_agg_mode="token-mean", global_batch_info={})
    loss_config = SimpleNamespace(
        loss_mode="k3",
        loss_max_clamp=None,
        use_policy_gradient=False,
        response_stream_weight=1.0,
        rejected_draft_stream_weight=1.0,
        confidence_loss_weight=confidence_loss_weight,
        confidence_tv_target_mix=confidence_tv_target_mix,
    )
    distillation_config = SimpleNamespace(distillation_loss=loss_config)
    return config, distillation_config, model_output, data, confidence_logits, confidence_labels, confidence_mask


def test_confidence_bce_added_to_distillation_loss_and_metrics():
    config, distillation_config, model_output, data, logits, labels, mask = _confidence_loss_inputs(1.0)

    loss, metrics = distillation_loss(config, distillation_config, model_output, data)

    base_loss = kl_penalty(torch.tensor([-0.4, -0.9]), torch.tensor([-0.5, -0.7]), "k3").mean()
    expected_bce = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], reduction="none").mean()
    assert torch.allclose(loss, base_loss + expected_bce)
    assert torch.allclose(
        torch.as_tensor(metrics["distillation/dspark_confidence_bce_loss"].values[0]),
        expected_bce,
    )
    assert torch.allclose(
        torch.as_tensor(metrics["distillation/dspark_accept_label_mean"].values[0]),
        torch.tensor(0.5),
    )
    assert torch.allclose(
        torch.as_tensor(metrics["distillation/dspark_confidence_pred_mean"].values[0]),
        logits[mask].sigmoid().mean(),
    )
    assert metrics["distillation/dspark_confidence_token_count"].values == [2.0]


def test_confidence_bce_weight_scales_and_zero_disables_term():
    base_loss = kl_penalty(torch.tensor([-0.4, -0.9]), torch.tensor([-0.5, -0.7]), "k3").mean()

    config, distillation_config, model_output, data, logits, labels, mask = _confidence_loss_inputs(0.5)
    loss, _ = distillation_loss(config, distillation_config, model_output, data)
    expected_bce = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], reduction="none").mean()
    assert torch.allclose(loss, base_loss + 0.5 * expected_bce)

    config, distillation_config, model_output, data, _, _, _ = _confidence_loss_inputs(0.0)
    loss, _ = distillation_loss(config, distillation_config, model_output, data)
    assert torch.allclose(loss, base_loss)


def test_confidence_tv_target_mix_raises():
    config, distillation_config, model_output, data, _, _, _ = _confidence_loss_inputs(1.0, 0.5)
    with pytest.raises(NotImplementedError):
        distillation_loss(config, distillation_config, model_output, data)


def test_confidence_stream_absent_is_noop_for_dflash():
    config, distillation_config, model_output, data, _, _, _ = _confidence_loss_inputs(1.0)
    for key in list(model_output):
        if "dspark" in key:
            del model_output[key]

    loss, metrics = distillation_loss(config, distillation_config, model_output, data)

    base_loss = kl_penalty(torch.tensor([-0.4, -0.9]), torch.tensor([-0.5, -0.7]), "k3").mean()
    assert torch.allclose(loss, base_loss)
    assert "distillation/dspark_confidence_bce_loss" not in metrics


class _TinyMainModel:
    """Plain-object frozen target: embeddings + lm_head + hidden_states output."""

    def __init__(self, vocab_size=16, hidden_size=8, num_layers=2):
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=num_layers)

    def eval(self):
        return self

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def __call__(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        hidden = self.embed_tokens(input_ids)
        hidden_states = tuple(hidden for _ in range(self.config.num_hidden_layers + 1))
        return SimpleNamespace(hidden_states=hidden_states)


class _TinyDsparkDraft:
    """Identity DSpark draft: draft_hidden == noise embedding, plus both heads."""

    def __init__(self, vocab_size=16, hidden_size=8, block_size=4, markov_rank=2, mask_token_id=15):
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.markov_head = StudentVanillaMarkovHead(vocab_size=vocab_size, markov_rank=markov_rank)
        self.confidence_head = torch.nn.Linear(hidden_size, 1)
        self.confidence_head_with_markov = False
        self.config = SimpleNamespace(
            _attn_implementation="eager",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dflash_config={
                "mask_token_id": mask_token_id,
                "target_layer_ids": [1],
                "markov_rank": markov_rank,
                "enable_confidence_head": True,
            },
        )

    def __call__(self, position_ids=None, attention_mask=None, noise_embedding=None, target_hidden=None, **kwargs):
        return noise_embedding


def _tiny_student(config=None):
    torch.manual_seed(0)
    student = _bare_student(config=config)
    student.main_model = _TinyMainModel()
    student.draft_model = _TinyDsparkDraft()
    student.target_layer_ids = [1]
    return student


def _tiny_opd_kwargs():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": None,
        "prompt_lengths": torch.tensor([3], dtype=torch.long),
        "response_lengths": torch.tensor([8], dtype=torch.long),
        "reject_token_indices": torch.tensor([[2]], dtype=torch.long),
        "rejected_draft_anchor_indices": torch.tensor([[0]], dtype=torch.long),
        "rejected_draft_offsets": torch.tensor([[2]], dtype=torch.long),
        "rejected_draft_token_ids": torch.tensor([[9]], dtype=torch.long),
        "rejected_draft_teacher_logprobs": torch.tensor([[-0.4]], dtype=torch.float32),
        "rejected_draft_mask": torch.tensor([[True]]),
    }


def _expected_logprob(student, prev_token_id, label_token_id):
    mask_embed = student.main_model.embed_tokens(
        torch.tensor([student.draft_model.mask_token_id], dtype=torch.long)
    )
    base_logits = student.main_model.lm_head(mask_embed)
    bias = student.draft_model.markov_head.compute_step_bias(torch.tensor([prev_token_id], dtype=torch.long))
    return torch.log_softmax((base_logits + bias).float(), dim=-1)[0, label_token_id]


def test_forward_opd_dspark_end_to_end_markov_and_confidence():
    student = _tiny_student()
    output = student._forward_opd(**_tiny_opd_kwargs())

    assert int(output["dflash_opd_draft_variant_id"].item()) == 1

    # Response stream, block anchored at seq position 2: draft position 1
    # predicts token 4 with prev token 3, and its logprob is stored at row 2.
    assert torch.allclose(output["dflash_log_probs"][0, 2], _expected_logprob(student, 3, 4))
    # Draft position 2 of the same block: prev token is the accepted token 4.
    assert torch.allclose(output["dflash_log_probs"][0, 3], _expected_logprob(student, 4, 5))
    # Rejected draft token 9 at offset 2 of the block anchored at seq position 3:
    # prev token is the last accepted token 5.
    assert torch.allclose(
        output["dflash_rejected_draft_student_log_probs"][0, 0],
        _expected_logprob(student, 5, 9),
    )
    assert torch.allclose(
        output["dflash_rejected_draft_teacher_log_probs"],
        torch.tensor([[-0.4]]),
    )
    assert output["dflash_rejected_draft_loss_mask"].tolist() == [[True]]

    # Confidence stream: offset=2 -> slot 1 accepted (1), slot 2 rejected (0).
    assert output["dflash_dspark_confidence_labels"].tolist() == [[[0.0, 1.0, 0.0, 0.0]]]
    assert output["dflash_dspark_confidence_mask"].tolist() == [[[False, True, True, False]]]
    mask_embed = student.main_model.embed_tokens(torch.tensor([15], dtype=torch.long))
    expected_confidence = student.draft_model.confidence_head(mask_embed).squeeze(-1).float()
    assert torch.allclose(output["dflash_dspark_confidence_logits"][0, 0, 1], expected_confidence[0])
    assert torch.allclose(output["dflash_dspark_confidence_logits"][0, 0, 2], expected_confidence[0])
    assert int(output["dflash_opd_dspark_confidence_token_count"].item()) == 2


def test_forward_opd_dflash_override_disables_dspark():
    config = PretrainedConfig()
    config.verl_dflash_draft_variant = "dflash"
    student = _tiny_student(config=config)
    output = student._forward_opd(**_tiny_opd_kwargs())

    assert "dflash_opd_draft_variant_id" not in output
    assert "dflash_dspark_confidence_logits" not in output

    # Without the Markov bias the response logprob uses only the base logits.
    mask_embed = student.main_model.embed_tokens(
        torch.tensor([student.draft_model.mask_token_id], dtype=torch.long)
    )
    base_log_probs = torch.log_softmax(student.main_model.lm_head(mask_embed).float(), dim=-1)
    assert torch.allclose(output["dflash_log_probs"][0, 2], base_log_probs[0, 4])
    assert torch.allclose(output["dflash_rejected_draft_student_log_probs"][0, 0], base_log_probs[0, 9])

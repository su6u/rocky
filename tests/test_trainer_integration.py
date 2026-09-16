"""Exercise real Trainer and PEFT plumbing with random tiny weights, never Rocky substitutes."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
datasets = pytest.importorskip("datasets")

from rocky.tokenization import ReplyCollator  # noqa: E402
from rocky.training import (  # noqa: E402
    PreferenceCollator,
    attach_reference,
    load_policy,
    preference_trainer_class,
    sft_trainer_class,
)


def native_gemma():
    # Random small native architecture fixture: PLE and shared KV are kept.
    # This does not load, replace, or approximate the deployed checkpoint.
    config = transformers.Gemma4Config(text_config={
        "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
        "num_attention_heads": 2, "num_key_value_heads": 1, "num_global_key_value_heads": 1,
        "head_dim": 16, "global_head_dim": 16, "vocab_size": 32,
        "vocab_size_per_layer_input": 32, "hidden_size_per_layer_input": 4,
        "num_kv_shared_layers": 2, "layer_types": ["sliding_attention", "full_attention"] * 2,
        "sliding_window": 8, "attention_k_eq_v": True,
        "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                            "full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
    }, vision_config=None, audio_config=None)
    return transformers.AutoModelForMultimodalLM.from_config(config)


def test_native_e4b_ple_shared_kv_and_adapter_backward(monkeypatch):
    base = native_gemma()
    monkeypatch.setattr(transformers.AutoModelForMultimodalLM, "from_pretrained", lambda *a, **kw: (base, {}))
    model = load_policy({"model_id": "test-fixture", "revision": "test", "lora_rank": 2, "lora_alpha": 4})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    from rocky.objectives import sequence_logps

    labels = torch.tensor([[-100, -100, 3, 4]])
    sums, counts = sequence_logps(model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits, labels)
    loss = -(sums / counts).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)


def test_e4b_shared_kv_cached_full_and_incremental_logits_match():
    model = native_gemma().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    with torch.no_grad():
        uncached = model(input_ids=ids, use_cache=False).logits
        cached = model(input_ids=ids, use_cache=True).logits
        # Cross the fixture's sliding-window boundary, not just a short prefix.
        prefix = model(input_ids=ids[:, :9], use_cache=True)
        for end in range(10, 13):
            prefix = model(input_ids=ids[:, end - 1:end], past_key_values=prefix.past_key_values, use_cache=True)
            torch.testing.assert_close(prefix.logits[:, -1], uncached[:, end - 1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(cached, uncached, atol=1e-6, rtol=1e-5)


def test_selected_logits_and_checkpointing_preserve_loss_and_gradients():
    from copy import deepcopy

    from rocky.objectives import reply_logps, sequence_logps

    base = native_gemma().train()
    selected = deepcopy(base)
    selected.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
    labels = torch.tensor([[-100, -100, -100, -100, 5, 6], [-100, -100, 7, 8, -100, -100]])
    sums, counts = sequence_logps(base(input_ids=inputs, attention_mask=mask, use_cache=False).logits, labels)
    dense_loss = -(sums / counts).mean()
    dense_loss.backward()
    projected = []
    hook = selected.lm_head.register_forward_pre_hook(lambda module, args: projected.append(args[0].shape[1]))
    sparse_sums, sparse_counts = reply_logps(selected, {"input_ids": inputs, "attention_mask": mask,
                                                      "labels": labels, "use_cache": False})
    sparse_loss = -(sparse_sums / sparse_counts).mean()
    sparse_loss.backward()
    hook.remove()
    assert projected == [4]  # Union of shifted target positions, not six full positions.
    torch.testing.assert_close(sparse_loss, dense_loss)
    for (name, left), (_, right) in zip(base.named_parameters(), selected.named_parameters()):
        if left.grad is not None:
            assert right.grad is not None, name
            torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-4, msg=name)


@pytest.mark.parametrize("objective", ["sft", "dpo", "dpo-norm", "simpo"])
def test_real_trainer_updates_and_saves_adapter(tmp_path, objective):
    torch.manual_seed(7)
    base = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        n_layer=1, n_head=2, n_embd=16, vocab_size=32, n_positions=32,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    ))
    model = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    chosen = {"input_ids": [1, 2, 3, 4], "attention_mask": [1] * 4, "labels": [-100, -100, 3, 4]}
    rejected = {"input_ids": [1, 2, 5], "attention_mask": [1] * 3, "labels": [-100, -100, 5]}
    args = transformers.TrainingArguments(output_dir=str(tmp_path), use_cpu=True, max_steps=1,
        per_device_train_batch_size=1, report_to="none", save_strategy="no", learning_rate=0.01,
        remove_unused_columns=False, prediction_loss_only=True, disable_tqdm=True)
    if objective == "sft":
        trainer = sft_trainer_class()(model=model, args=args, train_dataset=datasets.Dataset.from_list([chosen]),
            eval_dataset=datasets.Dataset.from_list([chosen]), data_collator=ReplyCollator(0))
    else:
        rows = [{"chosen": chosen, "rejected": rejected}]
        if objective != "simpo":
            attach_reference(model, rows, 0, "cpu")
            # Disabling the SFT adapter is intentionally not used for references.
            assert "reference_chosen" in rows[0]
        trainer = preference_trainer_class()(model=model, args=args, train_dataset=datasets.Dataset.from_list(rows),
            eval_dataset=datasets.Dataset.from_list(rows), data_collator=PreferenceCollator(0),
            objective=objective, beta=0.1, gamma=0.0)
    from rocky.tracking import RunTracking, tracking_callback

    tracker = RunTracking(tmp_path, mode="disabled", project="rocky-test", entity=None,
                          identity={}, diagnostics={}, hourly_usd=1)
    trainer.add_callback(tracking_callback(tracker))
    before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    result = trainer.train()
    assert result.global_step == 1
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters() if name in before)
    metrics = trainer.evaluate()
    assert "eval_loss" in metrics
    if objective != "sft":
        assert 0 <= metrics["eval_objective_win_rate"] <= 1
        assert metrics["eval_chosen_logp_per_token"] < 0
        assert metrics["eval_rejected_logp_per_token"] < 0
    trainer.save_model(str(tmp_path / "model"))
    assert (tmp_path / "model/adapter_model.safetensors").is_file()
    import json

    logged = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert any("eval/loss" in row for row in logged)
    assert all(row["trainer/step"] == 1 for row in logged)

"""Offline preference objectives. No reward model, sampling loop, or synthetic data."""

from __future__ import annotations


def token_logps(logits, targets):
    import torch

    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("logits and aligned targets differ")
    mask = targets != -100
    counts = mask.sum(-1)
    if torch.any(counts == 0):
        raise ValueError("empty preference completion")
    # Gemma has a 262k-token vocabulary. Never allocate FP32 softmax for masked
    # prompt positions: only the few spoken target positions need probabilities.
    positions = mask.nonzero(as_tuple=True)
    selected = -torch.nn.functional.cross_entropy(
        logits[positions].float(), targets[positions], reduction="none"
    )
    sums = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    sums = sums.index_add(0, positions[0], selected)
    if not torch.isfinite(sums).all():
        raise ValueError("nonfinite completion log probability")
    return sums, counts


def sequence_logps(logits, labels):
    return token_logps(logits[:, :-1], labels[:, 1:])


def reply_logps(model, batch):
    """Use native E4B output selection; never project masked history to vocabulary."""
    inputs = dict(batch)
    labels = inputs.pop("labels")
    if getattr(model.config, "model_type", None) == "gemma4":
        # A position predicts the following token. Union across the batch keeps
        # unequal chosen/rejected lengths and padding aligned without repacking.
        targets = labels[:, 1:]
        positions = (targets != -100).any(dim=0).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise ValueError("empty completion")
        logits = model(**inputs, logits_to_keep=positions).logits
        return token_logps(logits, targets[:, positions])
    return sequence_logps(model(**inputs).logits, labels)


def preference_loss(chosen, rejected, chosen_tokens, rejected_tokens, *, objective, beta, gamma=0.0,
                    reference_chosen=None, reference_rejected=None):
    import math

    import torch
    import torch.nn.functional as F

    if not math.isfinite(beta) or not math.isfinite(gamma) or beta <= 0 or gamma < 0:
        raise ValueError("invalid preference hyperparameters")
    if any(x.ndim != 1 or x.shape != chosen.shape for x in (chosen, rejected, chosen_tokens, rejected_tokens)):
        raise ValueError("preference scores and counts must be matching vectors")
    if torch.any(chosen_tokens <= 0) or torch.any(rejected_tokens <= 0):
        raise ValueError("preference token counts must be positive")
    if objective in {"dpo", "dpo-norm"}:
        if reference_chosen is None or reference_rejected is None:
            raise ValueError("DPO requires frozen SFT reference scores")
        if reference_chosen.shape != chosen.shape or reference_rejected.shape != chosen.shape:
            raise ValueError("reference score shape differs from policy")
        c = chosen - reference_chosen.detach()
        r = rejected - reference_rejected.detach()
        if objective == "dpo-norm":
            c, r = c / chosen_tokens, r / rejected_tokens
        margin = beta * (c - r)
    elif objective == "simpo":
        margin = beta * (chosen / chosen_tokens - rejected / rejected_tokens) - gamma
    else:
        raise ValueError("unknown preference objective")
    if not torch.isfinite(margin).all():
        raise ValueError("nonfinite preference margin")
    return -F.logsigmoid(margin)

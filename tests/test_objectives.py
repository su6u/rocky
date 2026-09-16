import pytest
import torch

from rocky.objectives import preference_loss, sequence_logps


def test_sequence_logps_counts_only_completion_tokens():
    logits = torch.zeros((1, 4, 3))
    labels = torch.tensor([[-100, 1, 2, -100]])
    sums, counts = sequence_logps(logits, labels)
    assert counts.tolist() == [2]
    assert torch.allclose(sums, torch.tensor([-2 * torch.log(torch.tensor(3.0))]))


def test_dpo_requires_fixed_reference_scores():
    values = dict(
        chosen=torch.tensor([-1.0]),
        rejected=torch.tensor([-2.0]),
        chosen_tokens=torch.tensor([1]),
        rejected_tokens=torch.tensor([1]),
        objective="dpo",
        beta=0.1,
    )
    with pytest.raises(ValueError, match="frozen SFT reference"):
        preference_loss(**values)


def test_all_supported_preference_objectives_are_finite():
    common = dict(
        chosen=torch.tensor([-1.0]),
        rejected=torch.tensor([-1.5]),
        chosen_tokens=torch.tensor([2]),
        rejected_tokens=torch.tensor([3]),
    )
    refs = dict(reference_chosen=torch.tensor([-1.2]), reference_rejected=torch.tensor([-1.4]))
    for objective, extra in (("dpo", refs), ("dpo-norm", refs), ("simpo", {})):
        loss = preference_loss(**common, objective=objective, beta=0.2, gamma=0.1, **extra)
        assert torch.isfinite(loss).all()

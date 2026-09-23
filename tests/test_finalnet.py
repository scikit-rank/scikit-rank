"""Tests for FINAL model components."""

import pytest
import torch
import torch.nn.functional as F

from scikit_rank import factories
from scikit_rank.factories import build_finalnet
from scikit_rank.modules.dcn import NumericEncoder
from scikit_rank.modules.finalnet import (
    FactorizedInteraction,
    FinalBlock,
    FinalNet,
    FinalNetConsistencyLoss,
    FinalNetFieldGate,
)
from scikit_rank.modules.losses import BCELoss
from scikit_rank.modules.reducers import Concat
from scikit_rank.run import TrainingModule


def test_factorized_interaction_sum_matches_formula() -> None:
    layer = FactorizedInteraction(2, 2, residual_type="sum", interaction_activation="identity")
    with torch.no_grad():
        layer._linear.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 2))
        layer._linear.bias.zero_()

    x = torch.tensor([[2.0, 3.0]])
    assert torch.equal(layer(x), torch.tensor([[6.0, 12.0]]))


def test_factorized_interaction_concat_matches_formula() -> None:
    layer = FactorizedInteraction(2, 4, residual_type="concat", interaction_activation="identity")
    with torch.no_grad():
        layer._linear.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 2))
        layer._linear.bias.zero_()

    x = torch.tensor([[2.0, 3.0]])
    assert torch.equal(layer(x), torch.tensor([[2.0, 3.0, 4.0, 9.0]]))


def test_concat_interaction_requires_even_output_width() -> None:
    with pytest.raises(ValueError, match="even output_dim"):
        FactorizedInteraction(2, 3)


def test_final_block_uses_reference_layer_order() -> None:
    block = FinalBlock(
        4,
        [6],
        hidden_activations="gelu",
        dropout=0.1,
        batch_norm=True,
    )
    assert isinstance(block._layers[0], FactorizedInteraction)
    assert isinstance(block._norms[0], torch.nn.BatchNorm1d)
    assert isinstance(block._activations[0], torch.nn.GELU)
    assert isinstance(block._dropouts[0], torch.nn.Dropout)
    assert block(torch.randn(3, 4)).shape == (3, 6)


def test_finalnet_averages_two_branch_logits() -> None:
    first_head = torch.nn.Linear(2, 1)
    second_head = torch.nn.Linear(2, 1)
    with torch.no_grad():
        first_head.weight.copy_(torch.tensor([[1.0, 0.0]]))
        first_head.bias.zero_()
        second_head.weight.copy_(torch.tensor([[0.0, 1.0]]))
        second_head.bias.zero_()
    model = FinalNet(
        layers={"num": NumericEncoder(2)},
        reducer=Concat(),
        block1=torch.nn.Identity(),
        head1=first_head,
        block2=torch.nn.Identity(),
        head2=second_head,
    )

    logits_a, logits_b = model.branch_logits({"num": torch.tensor([[2.0, 4.0]])})
    assert torch.equal(logits_a, torch.tensor([2.0]))
    assert torch.equal(logits_b, torch.tensor([4.0]))
    logits, branch_logits = model({"num": torch.tensor([[2.0, 4.0]])})
    assert torch.equal(logits, torch.tensor([3.0]))
    assert torch.equal(branch_logits, torch.tensor([[2.0, 4.0]]))


def test_finalnet_rejects_incomplete_second_branch() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        FinalNet(
            layers={"num": NumericEncoder(2)},
            reducer=Concat(),
            block1=torch.nn.Identity(),
            head1=torch.nn.Linear(2, 1),
            block2=torch.nn.Identity(),
        )


def test_finalnet_field_gate_matches_reference_field_axis_formula() -> None:
    gate = FinalNetFieldGate(input_dim=4, num_fields=2)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # Reference initialization makes the gated half an exact copy of the input.
    assert torch.equal(gate(x), torch.tensor([[1.0, 2.0, 3.0, 4.0] * 2]))

    with torch.no_grad():
        gate._linear.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        gate._linear.bias.copy_(torch.tensor([0.5, -0.5]))
    expected_gates = torch.tensor([[[7.5, 10.5], [14.5, 21.5]]])
    fields = x.reshape(1, 2, 2)
    expected = torch.cat((fields, fields * expected_gates), dim=1).flatten(start_dim=1)

    assert torch.equal(gate(x), expected)


def test_finalnet_field_gate_is_applied_only_to_first_branch() -> None:
    gate = FinalNetFieldGate(input_dim=2, num_fields=2)
    first_head = torch.nn.Linear(4, 1, bias=False)
    second_head = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        first_head.weight.copy_(torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
        second_head.weight.copy_(torch.tensor([[1.0, 0.0]]))
    model = FinalNet(
        layers={"num": NumericEncoder(2)},
        reducer=Concat(),
        block1=torch.nn.Identity(),
        head1=first_head,
        block2=torch.nn.Identity(),
        head2=second_head,
        field_gate=gate,
    )
    inputs = {"num": torch.tensor([[2.0, 4.0]])}

    first_before, second_before = model.branch_logits(inputs)
    with torch.no_grad():
        gate._linear.bias.fill_(2.0)
    first_after, second_after = model.branch_logits(inputs)

    assert torch.equal(first_before, torch.tensor([2.0]))
    assert torch.equal(first_after, torch.tensor([4.0]))
    assert torch.equal(second_before, second_after)


@pytest.mark.parametrize("use_pytorch_init", [False, True])
def test_finalnet_factory_restores_reference_gate_initialization(
    use_pytorch_init: bool,
) -> None:
    plain = build_finalnet(
        n_num_features=0,
        cardinalities=[5, 7],
        embedding_dims=[2, 2],
        block_type="1B",
        block1_hidden_units=[4],
        use_field_gate=False,
        use_pytorch_init=use_pytorch_init,
    )
    gated = build_finalnet(
        n_num_features=0,
        cardinalities=[5, 7],
        embedding_dims=[2, 2],
        block_type="1B",
        block1_hidden_units=[4],
        use_field_gate=True,
        use_pytorch_init=use_pytorch_init,
    )
    gate = gated.field_gate()

    assert gate is not None
    assert torch.count_nonzero(gate._linear.weight) == 0
    assert torch.equal(gate._linear.bias, torch.ones(2))
    # 16 extra first-layer weights plus the gate's 2x2 weights and two biases.
    assert (
        sum(p.numel() for p in gated.parameters()) - sum(p.numel() for p in plain.parameters())
        == 22
    )


@pytest.mark.parametrize("use_pytorch_init", [False, True])
def test_build_finalnet_initialization_policy(monkeypatch, use_pytorch_init: bool) -> None:
    initialized: list[torch.nn.Module] = []
    monkeypatch.setattr(factories, "_init_weights", initialized.append)

    model = factories.build_finalnet(
        n_num_features=1,
        cardinalities=[],
        block_type="1B",
        block1_hidden_units=[4],
        use_pytorch_init=use_pytorch_init,
    )

    assert initialized == ([] if use_pytorch_init else [model])


def test_finalnet_field_gate_rejects_unequal_field_widths() -> None:
    with pytest.raises(ValueError, match="equal-width field representations"):
        build_finalnet(
            n_num_features=0,
            cardinalities=[5, 7],
            embedding_dims=[2, 3],
            block_type="1B",
            block1_hidden_units=[4],
            use_field_gate=True,
        )


def test_finalnet_field_gate_supports_unified_embeddings_without_explicit_dims() -> None:
    model = build_finalnet(
        n_num_features=0,
        cardinalities=[5, 7],
        embedding_dims=None,
        cat_encoder="unified:embedding_dim=3",
        block_type="1B",
        block1_hidden_units=[4],
        use_field_gate=True,
    )

    assert model.field_gate() is not None


def _reference_finalnet_2b_loss(
    branch_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    detach_teacher: bool,
) -> torch.Tensor:
    mean_logits = branch_logits.mean(dim=1)
    teacher = mean_logits.sigmoid()
    if detach_teacher:
        teacher = teacher.detach()
    return (
        F.binary_cross_entropy_with_logits(mean_logits, target)
        + F.binary_cross_entropy_with_logits(branch_logits[:, 0], teacher)
        + F.binary_cross_entropy_with_logits(branch_logits[:, 1], teacher)
    )


def test_finalnet_2b_consistency_loss_matches_reference_and_detaches_teacher() -> None:
    first_head = torch.nn.Linear(2, 1, bias=False)
    second_head = torch.nn.Linear(2, 1, bias=False)
    initial = torch.tensor([[-1.0, 0.5], [2.0, -0.25]])
    with torch.no_grad():
        first_head.weight.copy_(initial[:, 0].unsqueeze(0))
        second_head.weight.copy_(initial[:, 1].unsqueeze(0))
    model = FinalNet(
        layers={"num": NumericEncoder(2)},
        reducer=Concat(),
        block1=torch.nn.Identity(),
        head1=first_head,
        block2=torch.nn.Identity(),
        head2=second_head,
    )
    base_loss = BCELoss()
    consistency_loss = FinalNetConsistencyLoss(base_loss)
    module = TrainingModule(model, consistency_loss).train()
    target = torch.tensor([0.0, 1.0])

    output = module({"num": torch.eye(2), "target": target})
    expected_logits = initial.clone().requires_grad_()
    expected = _reference_finalnet_2b_loss(expected_logits, target, detach_teacher=True)
    assert type(module) is TrainingModule
    assert torch.allclose(output["logits"], initial.mean(dim=1))
    assert torch.allclose(output["loss"], expected)

    output["loss"].backward()
    expected.backward()
    actual_grad = torch.stack(
        (first_head.weight.grad.squeeze(0), second_head.weight.grad.squeeze(0)),
        dim=1,
    )
    assert torch.allclose(actual_grad, expected_logits.grad)

    non_detached_logits = initial.clone().requires_grad_()
    _reference_finalnet_2b_loss(
        non_detached_logits,
        target,
        detach_teacher=False,
    ).backward()
    assert not torch.allclose(actual_grad, non_detached_logits.grad)

    assert consistency_loss.unwrap() is base_loss
    with pytest.raises(ValueError, match="exactly one branch-logits tensor"):
        consistency_loss(initial.mean(dim=1), target)

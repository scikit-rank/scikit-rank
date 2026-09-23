import numpy as np
import pytest
import tabm as tabm_lib
import torch

from scikit_rank import factories
from scikit_rank.factories import build_tabm
from scikit_rank.modules.losses import Loss, MSELoss
from scikit_rank.modules.tabm import TabMEnsembleLoss, aggregate_member_logits
from scikit_rank.run import TrainingModule


class _CustomNumericEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 5)

    def output_dim(self) -> int:
        return 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def _first_scaling(model: torch.nn.Module, arch_type: str) -> torch.Tensor:
    backbone = model.backbone()
    return backbone.blocks[0][0].r if arch_type == "tabm" else backbone.affine.weight


def _assert_chunk_initialization(scaling: torch.Tensor, chunks: list[int]) -> None:
    start = 0
    for chunk in chunks:
        values = scaling[:, start : start + chunk]
        torch.testing.assert_close(values, values[:, :1].expand_as(values))
        start += chunk
    assert start == scaling.size(1)
    boundaries = torch.tensor(chunks).cumsum(0)[:-1]
    assert all(
        torch.any(scaling[:, boundary - 1] != scaling[:, boundary]) for boundary in boundaries
    )


@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini"])
def test_build_tabm_supports_all_builtin_feature_streams(arch_type: str) -> None:
    bins = [torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([-2.0, 0.0, 2.0])]
    model = build_tabm(
        num_feature_names=["n1", "n2"],
        cat_feature_names=["c1", "c2"],
        cardinalities=[5, 7],
        embedding_dims=[3, 3],
        num_encoder="ple:embedding_dim=3",
        num_encoder_bins=bins,
        cat_encoder="unified:embedding_dim=3",
        multihash_feature_names=["mh1", "mh2"],
        multihash_encoder="multihash:cardinality=17;n_hashes=2;embedding_dim=2",
        multihash_n_hashes=2,
        embedding_encoders={"user": "tower:output_dim=4;dropout=0.0"},
        embedding_input_dims={"user": 6},
        n_blocks=2,
        d_block=8,
        k=3,
        arch_type=arch_type,
        aggregation="binary_probability",
    )
    inputs = {
        "num": torch.randn(4, 2),
        "cat": torch.stack([torch.randint(0, 5, (4,)), torch.randint(0, 7, (4,))], 1),
        "multihash": torch.randint(0, 17, (4, 4)),
        "user": torch.randn(4, 6),
    }

    assert model.forward_members(inputs).shape == (4, 3)
    logits, member_logits = model(inputs)
    assert logits.shape == (4,)
    assert member_logits.shape == (4, 3)
    _assert_chunk_initialization(
        _first_scaling(model, arch_type),
        [3, 3, 3, 3, 4, 4, 4],
    )


def test_custom_encoder_output_is_one_initialization_chunk() -> None:
    model = build_tabm(
        num_feature_names=["n1", "n2"],
        cat_feature_names=[],
        cardinalities=[],
        embedding_dims=[],
        num_encoder=_CustomNumericEncoder(),
        n_blocks=1,
        d_block=4,
        k=2,
        start_scaling_init="normal",
    )
    _assert_chunk_initialization(_first_scaling(model, "tabm"), [5])


def test_tabm_probability_aggregation_averages_member_probabilities() -> None:
    binary = torch.tensor([[-2.0, 0.5, 3.0], [1.0, -1.0, 0.0]])
    binary_score = aggregate_member_logits(binary, "binary_probability")
    torch.testing.assert_close(binary_score.sigmoid(), binary.sigmoid().mean(dim=1))

    multiclass = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]],
            [[-1.0, 0.5, 0.0], [3.0, -2.0, 1.0]],
        ],
    )
    multiclass_score = aggregate_member_logits(multiclass, "multiclass_probability")
    torch.testing.assert_close(
        multiclass_score.softmax(dim=-1),
        multiclass.softmax(dim=-1).mean(dim=1),
    )


def test_tabm_coral_head_has_member_and_aggregated_shapes() -> None:
    model = build_tabm(
        num_feature_names=["x"],
        cat_feature_names=[],
        cardinalities=[],
        embedding_dims=[],
        n_outputs=4,
        use_coral_head=True,
        aggregation="coral_probability",
        n_blocks=1,
        d_block=6,
        k=3,
        start_scaling_init="random-signs",
    )
    inputs = {"num": torch.randn(5, 1)}
    assert model.forward_members(inputs).shape == (5, 3, 3)
    logits, member_logits = model(inputs)
    assert logits.shape == (5, 3)
    assert member_logits.shape == (5, 3, 3)


def test_tabm_style_adapter_initialization_is_preserved() -> None:
    model = build_tabm(
        num_feature_names=["x1", "x2"],
        cat_feature_names=[],
        cardinalities=[],
        embedding_dims=[],
        n_blocks=2,
        d_block=5,
        k=4,
        start_scaling_init="random-signs",
    )
    first = model.backbone().blocks[0][0]
    second = model.backbone().blocks[1][0]
    assert isinstance(first, tabm_lib.LinearBatchEnsemble)
    assert torch.all(first.r.abs() == 1)
    assert torch.all(first.s == 1)
    assert torch.all(second.r == 1)
    assert torch.all(second.s == 1)


@pytest.mark.parametrize("use_pytorch_init", [False, True])
def test_build_tabm_initialization_policy(monkeypatch, use_pytorch_init: bool) -> None:
    initialized: list[torch.nn.Module] = []
    monkeypatch.setattr(factories, "_init_weights", initialized.append)

    model = factories.build_tabm(
        num_feature_names=["x"],
        cat_feature_names=[],
        cardinalities=[],
        embedding_dims=[],
        n_blocks=1,
        d_block=4,
        k=2,
        use_pytorch_init=use_pytorch_init,
    )

    assert initialized == ([] if use_pytorch_init else [model.layers()])


def test_tabm_ensemble_loss_averages_individual_losses() -> None:
    model = build_tabm(
        num_feature_names=["x"],
        cat_feature_names=[],
        cardinalities=[],
        embedding_dims=[],
        n_blocks=1,
        d_block=4,
        dropout=0.0,
        k=2,
        aggregation="mean",
    )
    loss = MSELoss()
    module = TrainingModule(model, TabMEnsembleLoss(loss))
    batch = {"num": torch.randn(3, 1), "target": torch.zeros(3)}
    members = model.forward_members(batch)
    output = module(batch)

    torch.testing.assert_close(output["logits"], model.aggregate_members(members))
    expected = torch.stack([loss(members[:, member], batch["target"]) for member in range(2)])
    torch.testing.assert_close(output["loss"], expected.mean())


class _RecordingGroupedLoss(Loss):
    requires_group = True

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Size, torch.Tensor]] = []

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        assert group is not None
        self.calls.append((scores.shape, group.detach().clone()))
        return (scores - target).square().mean()


def test_tabm_ensemble_loss_applies_grouped_loss_to_each_member() -> None:
    model = build_tabm(
        num_feature_names=["x"],
        cat_feature_names=[],
        cardinalities=[],
        embedding_dims=[],
        n_blocks=1,
        d_block=4,
        dropout=0.0,
        k=2,
        aggregation="mean",
    )
    loss = _RecordingGroupedLoss()
    module = TrainingModule(model, TabMEnsembleLoss(loss)).train()
    group = torch.tensor([0, 0, 1, 1])
    module({"num": torch.randn(4, 1), "target": torch.arange(4.0), "group": group})

    assert len(loss.calls) == 2
    assert all(shape == torch.Size([4]) for shape, _ in loss.calls)
    assert all(np.array_equal(seen.numpy(), group.numpy()) for _, seen in loss.calls)


def test_build_tabm_rejects_packed_architecture() -> None:
    with pytest.raises(ValueError, match="tabm-mini"):
        build_tabm(
            num_feature_names=["x"],
            cat_feature_names=[],
            cardinalities=[],
            embedding_dims=[],
            arch_type="tabm-packed",
        )

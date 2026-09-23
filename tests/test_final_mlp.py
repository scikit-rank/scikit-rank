import pytest
import torch

from scikit_rank import factories
from scikit_rank.factories import build_final_mlp
from scikit_rank.modules.final_mlp import InteractionAggregation


def test_interaction_aggregation_matches_multihead_bilinear_formula() -> None:
    torch.manual_seed(0)
    layer = InteractionAggregation(x_dim=6, y_dim=4, num_heads=2)
    x = torch.randn(5, 6)
    y = torch.randn(5, 4)

    head_x = x.reshape(5, 2, 3)
    head_y = y.reshape(5, 2, 2)
    expected = (
        layer._w_x(x).squeeze(-1)
        + layer._w_y(y).squeeze(-1)
        + torch.stack(
            [head_x[:, h] @ layer._w_xy[h] * head_y[:, h] for h in range(2)],
            dim=1,
        ).sum(dim=(1, 2))
    )
    torch.testing.assert_close(layer(x, y), expected)


def test_interaction_aggregation_pytorch_init_matches_linear() -> None:
    seed = 29
    torch.manual_seed(seed)
    torch.nn.Linear(6, 1)
    torch.nn.Linear(4, 1)
    expected = torch.nn.Linear(12, 1, bias=False).weight.detach()

    torch.manual_seed(seed)
    layer = InteractionAggregation(6, 4, num_heads=2, use_pytorch_init=True)

    torch.testing.assert_close(layer._w_xy.reshape(1, -1), expected)


@pytest.mark.parametrize("use_pytorch_init", [False, True])
def test_build_final_mlp_initialization_policy(monkeypatch, use_pytorch_init: bool) -> None:
    initialized: list[torch.nn.Module] = []
    monkeypatch.setattr(factories, "_init_weights", initialized.append)

    model = factories.build_final_mlp(
        num_feature_names=["x"],
        cat_feature_names=[],
        cardinalities=[],
        embedding_dims=[],
        mlp1_hidden_units=[4],
        mlp2_hidden_units=[4],
        use_pytorch_init=use_pytorch_init,
    )

    assert initialized == ([] if use_pytorch_init else [model])


def test_build_final_mlp_supports_all_builtin_feature_streams() -> None:
    bins = [torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([-2.0, 0.0, 2.0])]
    model = build_final_mlp(
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
        mlp1_hidden_units=[8],
        mlp2_hidden_units=[6],
        fs_hidden_units=[5],
        fs1_context=["n1", "c1", "mh1", "user"],
        fs2_context=[],
        num_heads=2,
        context_dim=3,
    )
    out = model(
        {
            "num": torch.randn(4, 2),
            "cat": torch.stack([torch.randint(0, 5, (4,)), torch.randint(0, 7, (4,))], 1),
            "multihash": torch.randint(0, 17, (4, 4)),
            "user": torch.randn(4, 6),
        },
    )
    assert out.shape == (4,)

    context1, context2 = model._feature_selection.context_layers()
    assert context1 is not None
    assert context2 is None
    main_params = {id(p) for p in model.layers().parameters()}
    context_params = {id(p) for p in context1.parameters()}
    assert main_params.isdisjoint(context_params)


def test_final_mlp_without_feature_selection_runs() -> None:
    model = build_final_mlp(
        num_feature_names=["x"],
        cat_feature_names=[],
        cardinalities=[],
        embedding_dims=[],
        mlp1_hidden_units=[4],
        mlp2_hidden_units=[4],
        use_fs=False,
        num_heads=2,
    )
    assert model({"num": torch.randn(3, 1)}).shape == (3,)
    assert model._feature_selection is None


def test_final_mlp_validates_context_and_head_dimensions() -> None:
    common = {
        "num_feature_names": ["x"],
        "cat_feature_names": [],
        "cardinalities": [],
        "embedding_dims": [],
        "mlp1_hidden_units": [5],
        "mlp2_hidden_units": [4],
    }
    with pytest.raises(ValueError, match="Unknown FinalMLP context"):
        build_final_mlp(**common, fs1_context=["missing"])
    with pytest.raises(ValueError, match="divisible by num_heads"):
        build_final_mlp(**common, num_heads=2)

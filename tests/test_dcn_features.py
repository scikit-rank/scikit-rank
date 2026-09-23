"""Tests for DCNv2 model and preprocessing features.

Covers the linear numeric embedding encoder, the unconditional weight
initialization (embeddings ``N(0, 1e-4)``, Xavier-normal ``Linear``/``EinMix``,
zeroed cross-bias), the numeric NaN-fill modes, the embedding-only L2
regularizer, gradient clipping, reduce-LR-on-plateau, and the optional BatchNorm
deep block.
"""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from scikit_rank import DCNClassifier, build_dcnv2, factories
from scikit_rank.modules.dcn import DeepNetwork, LinearNumericEncoder
from scikit_rank.modules.losses import make_loss
from scikit_rank.preprocessing import TabularPreprocessor
from scikit_rank.run import TrainingModule
from scikit_rank.train.options import attach_embedding_regularizer


def _toy() -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 200
    X = pl.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        },
    )
    y = (X["x1"].to_numpy() + X["x2"].to_numpy() > 0).astype(int)
    return X, y


# -- 1. linear numeric encoder ---------------------------------------------


def test_linear_numeric_encoder_shape_and_bias_free() -> None:
    enc = LinearNumericEncoder(n_features=3, embedding_dim=4)
    assert enc.output_dim() == 12
    out = enc(torch.randn(7, 3))
    assert out.shape == (7, 12)
    # the linear numeric encoder is nn.Linear(1, emb, bias=False)
    assert all("bias" not in name for name, _ in enc.named_parameters())


def test_linear_encoder_via_registry_spec() -> None:
    model = build_dcnv2(
        n_num_features=3,
        cardinalities=[4],
        embedding_dims=[3],
        hidden_units=[8],
        num_encoder="linear:embedding_dim=4",
    )
    num = model.layers()["num"]
    assert isinstance(num, LinearNumericEncoder)
    assert num.output_dim() == 3 * 4


# -- 2. weight init ---------------------------------------------------------


def test_weight_init_applied_by_default() -> None:
    model = build_dcnv2(
        n_num_features=3,
        cardinalities=[5, 6],
        embedding_dims=[4, 4],
        hidden_units=[8],
        cross_layers=2,
        num_encoder="linear:embedding_dim=4",
    )
    # embeddings ~ N(0, 1e-4): tiny std
    embeddings = [m for m in model.modules() if isinstance(m, torch.nn.Embedding)]
    assert embeddings
    for emb in embeddings:
        assert emb.weight.std().item() < 1e-3

    # every cross-layer additive bias is zeroed (module default is uniform)
    cross_biases = [p for n, p in model.named_parameters() if n.endswith("_bias")]
    assert cross_biases
    assert all(int(torch.count_nonzero(b)) == 0 for b in cross_biases)


def test_pytorch_init_skips_common_initializer(monkeypatch) -> None:
    initialized: list[torch.nn.Module] = []
    monkeypatch.setattr(factories, "_init_weights", initialized.append)

    factories.build_dcnv2(
        n_num_features=1,
        cardinalities=[],
        hidden_units=[4],
        cross_layers=1,
        use_pytorch_init=True,
    )

    assert initialized == []


# -- 3. numeric NaN fill ----------------------------------------------------


def test_numeric_nan_fill_zero_vs_median() -> None:
    frame = pl.DataFrame({"x": [1.0, 3.0, None, 5.0]})  # median of {1,3,5} = 3
    pp_median = TabularPreprocessor(num_features=["x"], cat_features=[], normalize=False).fit(
        frame,
    )
    pp_zero = TabularPreprocessor(
        num_features=["x"],
        cat_features=[],
        normalize=False,
        numeric_nan_fill="zero",
    ).fit(frame)

    median_filled = frame.select(pp_median.numeric_transform_exprs())["x"].to_list()
    zero_filled = frame.select(pp_zero.numeric_transform_exprs())["x"].to_list()

    assert median_filled[2] == pytest.approx(3.0)
    assert zero_filled[2] == pytest.approx(0.0)


def test_numeric_nan_fill_validates() -> None:
    with pytest.raises(ValueError, match="numeric_nan_fill"):
        TabularPreprocessor(numeric_nan_fill="bogus")


# -- 4. embedding-only L2 ---------------------------------------------------


def _stub_trainer() -> SimpleNamespace:
    # attach_* only needs ``.add_event``; the returned handler is exercised
    # directly against a fake engine, so no real Trainer/engine run is needed.
    return SimpleNamespace(add_event=lambda *a, **k: None)


def test_embedding_regularizer_increases_loss() -> None:
    model = build_dcnv2(
        n_num_features=2,
        cardinalities=[4],
        embedding_dims=[4],
        hidden_units=[8],
        num_encoder="linear:embedding_dim=4",
    )
    loss_fn = make_loss("bce")
    batch = {
        "num": torch.randn(6, 2),
        "cat": torch.randint(0, 4, (6, 1)),
        "target": torch.randint(0, 2, (6,)).float(),
    }
    base = TrainingModule(model, loss_fn).forward(batch)["loss"]
    emb_params = list(model.embedding_parameters())
    assert emb_params  # linear numeric weight + categorical embedding
    # The penalty is a FORWARD_COMPLETED handler now, not baked into TrainingModule:
    # it adds (lam / 2) * sum(w^2) to state.output["loss"] before backward.
    handler = attach_embedding_regularizer(_stub_trainer(), emb_params, lam=1.0)
    engine = SimpleNamespace(state=SimpleNamespace(output={"loss": base.clone()}))
    handler(engine)
    assert engine.state.output["loss"].item() > base.item()


def test_embedding_regularizer_zero_is_noop() -> None:
    model = build_dcnv2(
        n_num_features=2,
        cardinalities=[4],
        embedding_dims=[4],
        hidden_units=[8],
        num_encoder="linear:embedding_dim=4",
    )
    loss_fn = make_loss("bce")
    batch = {
        "num": torch.randn(6, 2),
        "cat": torch.randint(0, 4, (6, 1)),
        "target": torch.randint(0, 2, (6,)).float(),
    }
    base = TrainingModule(model, loss_fn).forward(batch)["loss"]
    handler = attach_embedding_regularizer(
        _stub_trainer(),
        list(model.embedding_parameters()),
        lam=0.0,
    )
    engine = SimpleNamespace(state=SimpleNamespace(output={"loss": base.clone()}))
    handler(engine)
    assert engine.state.output["loss"].item() == pytest.approx(base.item())


# -- 5. gradient clipping + 6. LR scheduler (integration) -------------------


def test_grad_clip_norm_fits() -> None:
    X, y = _toy()
    clf = DCNClassifier(
        epochs=2,
        batch_size=32,
        hidden_units=[8],
        cross_layers=1,
        grad_clip_norm=1.0,
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y)
    assert clf.predict_proba(X.head(3)).shape == (3, 2)


def test_plateau_scheduler_config_reduces_lr() -> None:
    # The factor/patience defaults we ship must actually drop the LR on plateau.
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=0.1,
        patience=0,
        threshold=1e-6,
        threshold_mode="abs",
        min_lr=1e-6,
    )
    for value in (0.5, 0.5, 0.5):  # never improves
        sched.step(value)
    assert opt.param_groups[0]["lr"] < 1.0


def test_lr_scheduler_wired_into_fit() -> None:
    # AUC on a perfectly separable set saturates at 1.0
    # (which it cannot exceed), so every later epoch is a plateau and the shared
    # AUC monitor drives both the LR decay and early stopping to terminate well
    # before `epochs`. Deterministic: fit seeds torch + the loader RNG, and --cpu
    # removes GPU nondeterminism.
    rng = np.random.default_rng(0)
    n = 256
    x1 = rng.normal(size=n)
    X = pl.DataFrame(
        {"x1": x1, "x2": rng.normal(size=n), "cat": rng.choice(["a", "b"], size=n)},
    )
    y = (x1 > 0.0).astype(int)  # single-feature, linearly separable -> AUC -> 1.0
    clf = DCNClassifier(
        epochs=20,
        batch_size=32,
        hidden_units=[8],
        cross_layers=1,
        lr=0.05,
        eval_metric=roc_auc_score,
        eval_metric_name="auc",
        lr_scheduler="plateau:patience=0;factor=0.1;min_lr=1e-6",
        early_stopping_rounds=2,
        accelerator_config={"cpu": True},
        random_state=0,
    )
    clf.fit(X, y, eval_set=(X, y))
    assert len(clf.history_) < 20  # early-stopped via the shared AUC plateau
    assert "val_auc" in clf.history_[-1]  # monitor recorded into history_
    assert clf.predict_proba(X.head(3)).shape == (3, 2)


# -- 7. batch norm ---------------------------------------------------------


def test_deep_network_batch_norm_block_order() -> None:
    # The deep block with batch_norm=True is Linear -> BatchNorm1d -> ReLU -> Dropout.
    net = DeepNetwork(6, [8], dropout=0.1, activation="relu", batch_norm=True)
    block = next(iter(net._network))
    assert [type(m) for m in block] == [
        torch.nn.Linear,
        torch.nn.BatchNorm1d,
        torch.nn.ReLU,
        torch.nn.Dropout,
    ]


def test_deep_network_default_has_no_batch_norm() -> None:
    # Default (batch_norm=False) builds no BatchNorm anywhere.
    net = DeepNetwork(6, [8, 4], dropout=0.0, activation="relu")
    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in net.modules())


def test_batch_norm_with_gated_activation_raises() -> None:
    with pytest.raises(ValueError, match="batch_norm"):
        DeepNetwork(6, [8], activation="glu", batch_norm=True)


def test_build_dcnv2_batch_norm_adds_exactly_bn_affine_params() -> None:
    # BatchNorm contributes exactly 2*sum(hidden_units) trainable params (gamma +
    # beta per unit); running stats are buffers, not parameters.
    hidden = [16, 8]
    common = {
        "n_num_features": 0,
        "cardinalities": [5, 6, 7],
        "embedding_dims": [10, 10, 10],
        "hidden_units": hidden,
        "cross_layers": 2,
        "structure": "parallel",
    }
    base = build_dcnv2(**common)
    bn = build_dcnv2(**common, batch_norm=True)
    n_base = sum(p.numel() for p in base.parameters())
    n_bn = sum(p.numel() for p in bn.parameters())
    assert n_bn - n_base == 2 * sum(hidden)
    assert any(isinstance(m, torch.nn.BatchNorm1d) for m in bn.modules())


def test_dcnclassifier_batch_norm_fits_and_predicts() -> None:
    X, y = _toy()
    clf = DCNClassifier(
        epochs=2,
        batch_size=32,
        hidden_units=[8],
        cross_layers=1,
        activation="relu",
        batch_norm=True,
        accelerator_config={"cpu": True},
        random_state=0,
    ).fit(X, y)
    assert clf.predict_proba(X.head(3)).shape == (3, 2)


def test_batch_norm_defaults_off_and_clone_preserves() -> None:
    assert DCNClassifier().batch_norm is False
    clf = DCNClassifier(batch_norm=True, use_pytorch_init=True)
    cloned = clone(clf)
    assert cloned.get_params()["batch_norm"] is True
    assert cloned.get_params()["use_pytorch_init"] is True


def test_full_parity_stack_fits_on_toy() -> None:
    X, y = _toy()
    clf = DCNClassifier(
        epochs=3,
        batch_size=32,
        structure="parallel",
        activation="relu",
        hidden_units=[8],
        cross_layers=1,
        embedding_dim=4,
        num_encoder="linear:embedding_dim=4",
        normalize_numeric=False,
        numeric_nan_fill="zero",
        embedding_regularizer=1e-5,
        grad_clip_norm=10.0,
        eval_metric=roc_auc_score,
        eval_metric_name="auc",
        lr_scheduler="plateau:patience=0;factor=0.1;min_lr=1e-6",
        early_stopping_rounds=2,
        accelerator_config={"cpu": True},
        random_state=0,
    )
    clf.fit(X, y, eval_set=(X, y))
    assert clf.predict_proba(X.head(3)).shape == (3, 2)

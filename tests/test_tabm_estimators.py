import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from scikit_rank import TabMClassifier, TabMRanker, TabMRegressor


def _data(n: int = 72) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(17)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    frame = pl.DataFrame(
        {"x1": x1, "x2": x2, "cat": rng.choice(["a", "b", "c"], size=n)},
    )
    return frame, (x1 + x2 > 0).astype(int)


def _params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "n_blocks": 1,
        "d_block": 8,
        "k": 3,
        "dropout": 0.0,
        "embedding_dim": 4,
        "num_encoder": "linear:embedding_dim=4",
        "epochs": 1,
        "batch_size": 24,
        "accelerator_config": {"cpu": True},
        "random_state": 0,
    }
    return params | overrides


@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini"])
def test_tabm_classifier_is_cloneable_and_predicts_probabilities(arch_type: str) -> None:
    X, y = _data()
    estimator = TabMClassifier(**_params(arch_type=arch_type, use_pytorch_init=True))
    cloned = clone(estimator)
    assert cloned.get_params()["k"] == 3
    assert cloned.get_params()["use_pytorch_init"] is True

    estimator.fit(X, y)
    proba = estimator.predict_proba(X.head(5))
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("loss", ["cross_entropy", "coral_layer"])
def test_tabm_classifier_supports_multiclass(loss: str) -> None:
    X, _ = _data()
    y = np.arange(len(X)) % 3
    estimator = TabMClassifier(**_params(loss=loss)).fit(X, y)
    proba = estimator.predict_proba(X.head(5))
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("loss", ["cross_entropy", "coral_layer"])
def test_tabm_classifier_supports_binary_multioutput_losses(loss: str) -> None:
    X, y = _data()
    proba = TabMClassifier(**_params(loss=loss)).fit(X, y).predict_proba(X.head(5))
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_tabm_regressor_ranker_and_save_load_share_estimator_api(tmp_path) -> None:
    X, y = _data()
    regression = TabMRegressor(**_params()).fit(X, y.astype(np.float32))
    expected = regression.predict(X.head(4))
    path = tmp_path / "tabm-regressor.pkl"
    regression.save(path)
    np.testing.assert_array_equal(TabMRegressor.load(path).predict(X.head(4)), expected)

    group = np.arange(len(y)) // 6
    ranker = TabMRanker(**_params(loss="listwise")).fit(X, y, group=group)
    assert ranker.predict(X.head(4)).shape == (4,)


def test_tabm_rejects_muon() -> None:
    X, y = _data()
    with pytest.raises(ValueError, match="does not support optimizer='muon'"):
        TabMClassifier(**_params(optimizer="muon")).fit(X, y)

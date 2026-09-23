import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from scikit_rank import FinalMLPClassifier, FinalMLPRanker, FinalMLPRegressor


def _data(n: int = 96) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    frame = pl.DataFrame(
        {"x1": x1, "x2": x2, "cat": rng.choice(["a", "b", "c"], size=n)},
    )
    return frame, (x1 + x2 > 0).astype(int)


def _params() -> dict:
    return {
        "embedding_dim": 4,
        "num_encoder": "linear:embedding_dim=4",
        "mlp1_hidden_units": (8,),
        "mlp2_hidden_units": (8,),
        "fs_hidden_units": (4,),
        "fs1_context": ("x1",),
        "fs2_context": ("cat",),
        "num_heads": 2,
        "epochs": 1,
        "batch_size": 32,
        "accelerator_config": {"cpu": True},
        "random_state": 0,
    }


def test_final_mlp_classifier_is_cloneable_and_predicts_probabilities() -> None:
    X, y = _data()
    estimator = FinalMLPClassifier(**_params(), use_pytorch_init=True)
    cloned = clone(estimator)
    assert cloned.get_params()["fs1_context"] == ("x1",)
    assert cloned.get_params()["use_pytorch_init"] is True

    estimator.fit(X, y)
    proba = estimator.predict_proba(X.head(5))
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_final_mlp_classifier_predicts_from_a_lazy_frame() -> None:
    X, y = _data()
    estimator = FinalMLPClassifier(**_params()).fit(X, y)

    assert estimator.predict_proba(X.lazy()).shape == (len(X), 2)


def test_final_mlp_classifier_rejects_multiclass_and_multioutput_losses() -> None:
    X, y = _data()
    with pytest.raises(ValueError, match="binary classification only"):
        FinalMLPClassifier(**_params()).fit(X, np.arange(len(y)) % 3)
    with pytest.raises(TypeError, match="scalar-output loss"):
        FinalMLPClassifier(**_params(), loss="coral_layer").fit(X, y)


def test_final_mlp_regressor_and_ranker_share_estimator_api() -> None:
    X, y = _data()
    regression = FinalMLPRegressor(**_params()).fit(X, y.astype(np.float32))
    assert regression.predict(X.head(4)).shape == (4,)

    group = np.arange(len(y)) // 8
    ranker = FinalMLPRanker(**_params(), loss="listwise").fit(X, y, group=group)
    assert ranker.predict(X.head(4)).shape == (4,)

"""Lazy (streamed Arrow) training must match eager (in-memory) training.

The lazy and eager estimator paths use different batch sources
(:class:`LazyArrowBatchSource` vs :class:`TensorDatasetSource`). They are
numerically equivalent: fed the same data they must produce identical weights.
The only legitimate difference under mini-batch SGD is shuffle ordering, so
parity is asserted with an order-independent full-batch gradient
(``batch_size >= n`` in a single chunk), where both paths take the same steps.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from scikit_rank import DCNClassifier, DCNRegressor


def _frame(n: int = 256) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    score = 1.5 * x1 - 1.0 * x2 + np.select(
        [cat == "a", cat == "b", cat == "c"],
        [0.6, -0.3, 0.1],
    )
    return pl.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "cat": cat,
            "target": (score > 0).astype(np.int64),
            "reg_target": score.astype(np.float32),
        },
    )


def test_classifier_lazy_matches_eager_full_batch() -> None:
    df = _frame()
    params = {
        "epochs": 3,
        "batch_size": len(df),  # full-batch => order-independent gradient
        "chunk_rows": len(df) + 1,  # single Arrow record batch
        "hidden_units": [12],
        "cross_layers": 1,
        "num_features": ["x1", "x2"],
        "cat_features": ["cat"],
        "accelerator_config": {"cpu": True},
        "random_state": 0,
    }
    x_pred = df.select("x1", "x2", "cat")

    eager = DCNClassifier(**params).fit(x_pred, df["target"].to_numpy())
    lazy = DCNClassifier(**params).fit(df.lazy(), y="target")

    np.testing.assert_allclose(
        eager.predict_proba(x_pred),
        lazy.predict_proba(x_pred.lazy()),
        rtol=0,
        atol=1e-6,
    )


def test_regressor_lazy_matches_eager_full_batch() -> None:
    df = _frame()
    params = {
        "epochs": 3,
        "batch_size": len(df),
        "chunk_rows": len(df) + 1,
        "hidden_units": [10],
        "cross_layers": 1,
        "num_features": ["x1", "x2"],
        "cat_features": ["cat"],
        "accelerator_config": {"cpu": True},
        "random_state": 1,
    }
    x_pred = df.select("x1", "x2", "cat")

    eager = DCNRegressor(**params).fit(x_pred, df["reg_target"].to_numpy())
    lazy = DCNRegressor(**params).fit(df.lazy(), y="reg_target")

    # Continuous targets differ only by float32 summation order within the
    # full batch (~1e-7), so compare at float precision rather than bit-exact.
    np.testing.assert_allclose(
        eager.predict(x_pred),
        lazy.predict(x_pred.lazy()),
        rtol=0,
        atol=1e-6,
    )

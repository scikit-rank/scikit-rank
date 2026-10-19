# scikit-rank

**sklearn-compatible neural estimators for tabular ranking, classification, and regression.**

`scikit-rank` provides neural, drop-in alternatives to LightGBM / CatBoost / XGBoost
for tabular problems. Its model zoo pairs modern tabular backbones with a consistent
scikit-learn `fit` / `predict` / `predict_proba` API, so estimators work in existing
sklearn pipelines and with `GridSearchCV`, `clone`, `get_params`, and `set_params`.

```python
from scikit_rank import DCNClassifier, FinalNetClassifier, TabMClassifier
```

## Paper

[Scikit-Rank: Scikit-learn-Compatible Neural Ranking Models for Tabular Recommender Systems](https://dl.acm.org/doi/full/10.1145/3773078.3841291) was accepted to RecSys 2026.

## Model zoo

Every backbone has classifier, regressor, and learning-to-rank estimator variants.
Classifiers expose `predict_proba`; rankers train and score by query when used with a
group-aware loss. All estimators accept the common feature, training, evaluation, and
runtime options described below.

| Backbone | sklearn estimators | Architecture options |
|---|---|---|
| [DCNv2](https://arxiv.org/abs/2008.13535) | `DCNClassifier`, `DCNRegressor`, `DCNRanker` | `hidden_units`, `cross_layers`, `cross_rank`, `structure`, `cross_type` |
| [FinalNet](https://dl.acm.org/doi/epdf/10.1145/3539618.3591988) | `FinalNetClassifier`, `FinalNetRegressor`, `FinalNetRanker` | `block_type`, `block1_hidden_units`, `block2_hidden_units`, `use_field_gate`, `use_2b_consistency_loss` |
| [FinalMLP](https://arxiv.org/abs/2304.00902) | `FinalMLPClassifier`, `FinalMLPRegressor`, `FinalMLPRanker` | `mlp1_hidden_units`, `mlp2_hidden_units`, `use_fs`, `fs_hidden_units`, `fs1_context`, `fs2_context`, `num_heads` |
| [TabM](https://arxiv.org/pdf/2410.24210) | `TabMClassifier`, `TabMRegressor`, `TabMRanker` | `arch_type`, `n_blocks`, `d_block`, `dropout`, `k`, `start_scaling_init` |
| [DESTINE](https://arxiv.org/abs/2101.03654) | `DESTINEClassifier`, `DESTINERegressor`, `DESTINERanker` | `embedding_dim`, `attention_dim`, `num_heads`, `attention_layers`, `unary_mode`, `residual_mode`, `use_wide` |

`FinalMLPClassifier` currently supports binary classification only. The other
classifiers support binary and multiclass targets.

## Features

- **Five backbones, one estimator API.** Choose DCNv2, FinalNet, FinalMLP, TabM, or
  DESTINE without changing the surrounding sklearn workflow.
- **sklearn-native ergonomics.** Flat `__init__` hyperparameters, `fit(X, y, group=..., eval_set=...)`,
  early stopping, custom eval metrics, and full `get_params` / `set_params` / `clone`
  support — mirroring the LightGBM / XGBoost sklearn wrappers.
- **Flexible inputs.** Accepts `numpy.ndarray`, `pandas.DataFrame`, `polars.DataFrame`,
  and `polars.LazyFrame`. String categoricals and NaNs are handled natively (no manual
  encoding required). When `X` is a polars (Lazy)Frame, `y` and `group` may be passed as
  column names.
- **Out-of-core lazy training.** A `polars.LazyFrame` is preprocessed and streamed
  through a temporary Arrow IPC file, so datasets larger than memory can be trained
  without materializing the full frame. Lazy training is numerically equivalent to the
  in-memory path (see `tests/test_lazy_eager_parity.py`).
- **Rich feature encoders.** Per-feature and hashed (Unified Embedding / multihash)
  categorical encoders, piecewise-linear (PLE) and quantile numeric encoders, and
  named external embedding streams (e.g. precomputed entity vectors).
- **Ranking losses.** BCE, BPR, LambdaRank, listwise/softmax, ordinal (CORAL), and more —
  selectable by string spec, e.g. `loss="bpr:sampling=all_pairs"`.
- **Scales with 🤗 Accelerate.** Mixed precision, gradient accumulation/clipping, EMA
  weights, LR schedulers, and multi-GPU / DDP training (including sharded, group-aware
  ranking evaluation).

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) and targets Python ≥ 3.12.

```bash
uv sync            # full environment (torch, accelerate, scikit-rank)
```

Or add it to your own project:

```bash
uv add scikit-rank
```

## Quick start

### Classification

```python
import polars as pl
from scikit_rank import DCNClassifier

X = pl.DataFrame({"num": [0.1, 1.2, -0.3], "cat": ["a", "b", "a"]})
y = [0, 1, 0]

clf = DCNClassifier(
    hidden_units=[256, 128],
    cross_layers=3,
    epochs=10,
    batch_size=1024,
    num_features=["num"],
    cat_features=["cat"],
    random_state=0,
)
clf.fit(X, y)

proba = clf.predict_proba(X)   # (n_samples, n_classes)
labels = clf.predict(X)
```

### Choose a backbone

All backbones use the same task-oriented API. Replace the estimator class while
keeping the `fit` and prediction calls unchanged:

```python
from scikit_rank import (
    DESTINEClassifier,
    FinalMLPClassifier,
    FinalNetClassifier,
    TabMClassifier,
)

models = [
    FinalNetClassifier(block_type="2B", epochs=10),
    FinalMLPClassifier(use_fs=True, num_heads=1, epochs=10),
    TabMClassifier(arch_type="tabm", k=32, epochs=10),
    DESTINEClassifier(attention_layers=2, num_heads=2, epochs=10),
]

for model in models:
    model.fit(X, y)
    probabilities = model.predict_proba(X)
```

### Regression

```python
from scikit_rank import FinalNetRegressor

reg = FinalNetRegressor(
    block_type="2B",
    epochs=10,
    num_features=["num"],
    cat_features=["cat"],
).fit(X, y_reg)
pred = reg.predict(X)
```

### Ranking

Pass a `group` (query / impression id) so the loss and metrics are computed per group.
`group` can be an array or, for (Lazy)Frames, a column name.

```python
from scikit_rank import TabMRanker

ranker = TabMRanker(
    loss="listwise",
    n_blocks=3,
    d_block=256,
    k=32,
    epochs=10,
    num_features=["num"],
    cat_features=["cat"],
    random_state=0,
)
ranker.fit(train_df, y="click", group="impression_id")
scores = ranker.predict(test_df)   # higher score = ranked higher
```

### Lazy / out-of-core training

```python
lf = pl.scan_parquet("large_dataset.parquet")   # never fully materialized
clf = DCNClassifier(
    num_features=["x1", "x2"],
    cat_features=["cat"],
    chunk_rows=100_000,          # Arrow streaming chunk size
    epochs=5,
).fit(lf, y="target")
```

### Custom evaluation metric + early stopping

```python
from sklearn.metrics import roc_auc_score

clf = DCNClassifier(
    epochs=100,
    early_stopping_rounds=5,
    eval_metric=lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
    eval_metric_name="auc",
    eval_metric_direction="max",
)
clf.fit(X_train, y_train, eval_set=(X_val, y_val))
print(clf.history_)   # per-epoch train/val metrics
```

## Demo notebook

[`demo.ipynb`](demo.ipynb) is an interactive Jupyter notebook that showcases the
library's capabilities end to end. It demonstrates preparing tabular data,
configuring and training the sklearn-compatible DCNv2 estimators, and inspecting
their predictions and fitted state. Run it after installing the project
environment:

```bash
uv sync
uv run jupyter lab demo.ipynb
```

## Key hyperparameters

| Group | Parameters |
|---|---|
| Backbone architecture | Use the model-specific options in the [model zoo](#model-zoo), e.g. DCNv2 `cross_layers`, FinalNet `block_type`, FinalMLP `use_fs`, TabM `k`, or DESTINE `attention_layers` |
| Embeddings / encoders | `embedding_dim`, `cat_encoder`, `num_encoder`, `multihash_features`, `embedding_features`, `normalize_numeric`, `ple_n_bins` |
| Optimization | `loss`, `lr`, `weight_decay`, `optimizer`, `epochs`, `batch_size`, `lr_scheduler`, `grad_clip_norm`, `embedding_regularizer`, `ema_decay` |
| Evaluation | `eval_metric`, `eval_metric_name`, `eval_metric_direction`, `eval_metric_group_aware`, `early_stopping_rounds` |
| Data / runtime | `num_features`, `cat_features`, `chunk_rows`, `random_state`, `accelerator_config`, `verbose` |

Multi-GPU / device settings are passed through `accelerator_config` (e.g.
`accelerator_config={"cpu": True}` or mixed-precision / DDP options from 🤗 Accelerate).

## Experiments & benchmarks

The [`exps/`](exps/) directory contains reproducible benchmarks and the scripts,
configs, and hyperparameters behind the published results:

- **[`exps/bars/`](exps/bars/README.md)** — BARS-CTR benchmark: DCNv2 vs.
  LightGBM / CatBoost / XGBoost / FuxiCTR on Criteo and Avazu.
- **[`exps/temporal_article/`](exps/temporal_article/README.md)** — temporal-split
  experiments (Avazu-Time, MIND-small) with the best models and their exact
  hyperparameters from the paper.

See [`exps/README.md`](exps/README.md) for dataset sources and how to recreate the
temporal-split datasets.

## Citation

```bibtex
@inbook{scikitrank2026,
    author = {Milogradskii, Aleksandr and Veselov, Ilya and Klyukin, Yaroslav and Kurdun, Maryia and Lashinin, Oleg and Liakhnovich, Kiryl},
    title = {Scikit-Rank: Scikit-learn-Compatible Neural Ranking Models for Tabular Recommender Systems},
    year = {2026},
    isbn = {9798400722844},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3773078.3841291},
    booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
    pages = {1826–1828},
    numpages = {3}
}
```

## Development

If you have [devenv](https://devenv.sh/) support, run `devenv shell` from the
repository root to instantiate the development environment with the required
modules and tools.

```bash
devenv shell                 # optional: enter the configured development shell
uv sync                      # install project and development dependencies
uv run python -m pytest -q   # run the test suite
uv run ruff check            # lint
```

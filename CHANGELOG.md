# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-09-23

### Added

- A stable scikit-learn-compatible estimator API for tabular classification,
  regression, and learning-to-rank: `fit`, `predict`, `predict_proba`, cloning,
  parameter inspection, grid search, evaluation sets, and early stopping.
- Five neural tabular backbones, each with classifier, regressor, and ranker
  estimators: DCNv2, FinalNet, FinalMLP, TabM, and DESTINE.
- Input support for NumPy arrays, pandas DataFrames, polars DataFrames, and
  polars LazyFrames, including native string categoricals, missing values, and
  named target and group columns for polars inputs.
- Out-of-core LazyFrame training through streamed Arrow IPC materialization.
- Configurable categorical, numeric, multihash, piecewise-linear, quantile, and
  external embedding feature encoders.
- Ranking-loss support including BCE, BPR, LambdaRank, listwise/softmax, and
  ordinal (CORAL) objectives, with group-aware training and evaluation.
- Accelerate-powered training features: mixed precision, gradient accumulation
  and clipping, learning-rate schedulers, EMA weights, and multi-GPU/DDP
  execution.
- Training callbacks through `training_options`, plus public access to the
  training run's accelerator, model, optimizer, loaders, trainer, and scheduler
  for custom Ignite handlers, metrics, and diagnostics.
- Optimizer support including Muon, along with reproducible BARS-CTR benchmark
  configurations for the model zoo.

### Changed

- Neural modules now use PyTorch's default parameter initialization where
  applicable.
- The project includes CI and tag-driven PyPI/GitHub Release publishing; release
  tags must match the package version.

### Compatibility

- Requires Python 3.12 or 3.13.
- `FinalMLPClassifier` supports binary classification; the other classifiers
  support binary and multiclass targets.

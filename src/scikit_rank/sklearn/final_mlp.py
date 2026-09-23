"""Sklearn-compatible FinalMLP estimators."""

from __future__ import annotations
import contextlib
import copy
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import polars as pl
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from scikit_rank.data import to_polars
from scikit_rank.factories import (
    build_final_mlp,
    build_lr_scheduler_config,
    multihash_encoder_config,
)
from scikit_rank.modules.losses import (
    LOSSES,
    CORALLayerLoss,
    CrossEntropyLoss,
    Loss,
    make_loss,
)
from scikit_rank.preprocessing import TabularPreprocessor
from scikit_rank.run import TrainingRun
from scikit_rank.sklearn._classification import ClassificationTarget, class_probabilities
from scikit_rank.sklearn._data_router import DataRouter
from scikit_rank.sklearn._inference import score_tabular_model
from scikit_rank.sklearn._input_validation import validate_X, validate_y
from scikit_rank.train.optimizers import OptimizerConfig
from scikit_rank.utils import ModuleParserSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sklearn.utils import Tags

    from scikit_rank.sklearn._types import EvalSet, GroupLike, XLike, YLike
    from scikit_rank.train.options import TrainingOptions


class FinalMLPBase(BaseEstimator):
    """Shared fit/predict machinery for the FinalMLP estimators.

    Do not instantiate this class directly. Use :class:`FinalMLPClassifier`,
    :class:`FinalMLPRegressor`, or :class:`FinalMLPRanker`. The estimators expose
    explicit constructor parameters and support sklearn ``clone``,
    ``get_params``/``set_params``, and ``GridSearchCV``.

    Parameters
    ----------
    embedding_dim : int or None, default=10
        Width of categorical embeddings and feature-selection context vectors.
    mlp1_hidden_units, mlp2_hidden_units : sequence of int, default=(64, 64, 64)
        Hidden-layer widths of the two FinalMLP towers.
    mlp1_hidden_activations, mlp2_hidden_activations : str, default="relu"
        Activation applied to every hidden layer in the corresponding tower.
    mlp1_dropout, mlp2_dropout : float, default=0.0
        Dropout probability for the corresponding tower.
    mlp1_batch_norm, mlp2_batch_norm : bool, default=False
        Apply batch normalization in the corresponding tower.
    use_fs : bool, default=True
        Enable FinalMLP's two feature-selection gates.
    fs_hidden_units : sequence of int, default=(64,)
        Hidden-layer widths of the feature-selection networks.
    fs1_context, fs2_context : sequence of str, default=()
        Feature names used as context by each feature-selection gate. An empty
        sequence uses the learned context bias from the reference architecture.
    num_heads : int, default=1
        Number of heads in the bilinear fusion module.
    num_encoder, cat_encoder : str or torch.nn.Module
        Numeric and categorical encoder specifications. ``num_encoder="ple"``
        enables piecewise-linear numeric encoding.
    loss : str or Loss or None, default=None
        Loss specification or instance. ``None`` selects ``bce`` for binary
        classification, ``mse`` for regression, and ``lambdarank`` for ranking.
    lr, weight_decay, optimizer, optimizer_kwargs
        Optimizer configuration. ``optimizer`` defaults to ``"adamw"``.
    epochs : int, default=10
        Maximum training epochs.
    batch_size : int, default=1024
        Mini-batch size. Ranking batches preserve query boundaries.
    early_stopping_rounds, eval_metric, eval_metric_name,
    eval_metric_direction, eval_metric_group_aware
        Validation and early-stopping configuration. Pass ``eval_set`` to
        :meth:`fit` when early stopping is enabled.
    num_features, cat_features : sequence of str or None, default=None
        Explicit feature columns. ``None`` infers columns from input dtypes.
    multihash_features, multihash_encoder
        High-cardinality columns and their shared hashed encoder.
    embedding_features, embedding_encoders
        Named columns containing external embedding vectors and their encoders.
    normalize_numeric, n_quantiles, numeric_nan_fill, ple_n_bins
        Numeric preprocessing and PLE-bin configuration.
    lr_scheduler, grad_clip_norm, embedding_regularizer, ema_decay
        Optional scheduler, optimization, and weight-averaging controls.
    chunk_rows : int, default=100_000
        Streaming chunk size for lazy training and inference.
    random_state : int or None, default=None
        Torch and NumPy random seed.
    accelerator_config : dict or None, default=None
        Keyword arguments for :class:`accelerate.Accelerator`.
    verbose : bool, default=False
        Print training progress and epoch logs.

    Attributes
    ----------
    model_ : torch.nn.Module
        Fitted FinalMLP network, retained on CPU for stable pickling.
    loss_ : Loss
        Instantiated training loss.
    history_ : list[dict[str, float]]
        Per-epoch train and validation metrics.
    preprocessor_ : TabularPreprocessor
        Fitted feature preprocessor.
    n_features_in_ : int
        Number of fitted input features.
    feature_names_in_ : numpy.ndarray
        Input feature names in preprocessing order.

    """

    def __init__(  # noqa: PLR0913 -- sklearn requires explicit flat hyperparameters
        self,
        *,
        embedding_dim: int | None = 10,
        mlp1_hidden_units: list[int] | tuple[int, ...] = (64, 64, 64),
        mlp1_hidden_activations: str = "relu",
        mlp1_dropout: float = 0.0,
        mlp1_batch_norm: bool = False,
        mlp2_hidden_units: list[int] | tuple[int, ...] = (64, 64, 64),
        mlp2_hidden_activations: str = "relu",
        mlp2_dropout: float = 0.0,
        mlp2_batch_norm: bool = False,
        use_fs: bool = True,
        fs_hidden_units: list[int] | tuple[int, ...] = (64,),
        fs1_context: list[str] | tuple[str, ...] = (),
        fs2_context: list[str] | tuple[str, ...] = (),
        num_heads: int = 1,
        use_pytorch_init: bool = False,
        num_encoder: str | torch.nn.Module = "identity",
        cat_encoder: str | torch.nn.Module = "per_feature",
        loss: str | Loss | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        epochs: int = 10,
        batch_size: int = 1024,
        early_stopping_rounds: int | None = None,
        eval_metric: Callable[..., float] | None = None,
        eval_metric_name: str = "metric",
        eval_metric_direction: str = "max",
        eval_metric_group_aware: bool = False,
        num_features: Sequence[str] | None = None,
        cat_features: Sequence[str] | None = None,
        multihash_features: Sequence[str] | None = None,
        multihash_encoder: str | torch.nn.Module = "multihash",
        embedding_features: dict[str, str] | None = None,
        embedding_encoders: dict[str, str | torch.nn.Module] | None = None,
        normalize_numeric: bool | str | None = True,
        n_quantiles: int = 1000,
        numeric_nan_fill: Literal["median", "zero"] = "median",
        ple_n_bins: int = 42,
        lr_scheduler: str | None = None,
        grad_clip_norm: float | None = None,
        embedding_regularizer: float = 0.0,
        ema_decay: float | None = None,
        chunk_rows: int = 100_000,
        random_state: int | None = None,
        accelerator_config: dict[str, Any] | None = None,
        training_options: TrainingOptions | None = None,
        verbose: bool = False,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_encoder = num_encoder
        self.cat_encoder = cat_encoder
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.eval_metric_name = eval_metric_name
        self.eval_metric_direction = eval_metric_direction
        self.eval_metric_group_aware = eval_metric_group_aware
        self.num_features = num_features
        self.cat_features = cat_features
        self.multihash_features = multihash_features
        self.multihash_encoder = multihash_encoder
        self.embedding_features = embedding_features
        self.embedding_encoders = embedding_encoders
        self.normalize_numeric = normalize_numeric
        self.n_quantiles = n_quantiles
        self.numeric_nan_fill = numeric_nan_fill
        self.ple_n_bins = ple_n_bins
        self.lr_scheduler = lr_scheduler
        self.grad_clip_norm = grad_clip_norm
        self.embedding_regularizer = embedding_regularizer
        self.ema_decay = ema_decay
        self.chunk_rows = chunk_rows
        self.random_state = random_state
        self.accelerator_config = accelerator_config
        self.training_options = training_options
        self.verbose = verbose
        self.mlp1_hidden_units = mlp1_hidden_units
        self.mlp1_hidden_activations = mlp1_hidden_activations
        self.mlp1_dropout = mlp1_dropout
        self.mlp1_batch_norm = mlp1_batch_norm
        self.mlp2_hidden_units = mlp2_hidden_units
        self.mlp2_hidden_activations = mlp2_hidden_activations
        self.mlp2_dropout = mlp2_dropout
        self.mlp2_batch_norm = mlp2_batch_norm
        self.use_fs = use_fs
        self.fs_hidden_units = fs_hidden_units
        self.fs1_context = fs1_context
        self.fs2_context = fs2_context
        self.num_heads = num_heads
        self.use_pytorch_init = use_pytorch_init

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        tags.input_tags.sparse = False
        return tags

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        """Encode the raw target into the float32 array seen by the loss."""
        return y.astype(np.float32)

    def _y_expr(self, target_col: str) -> pl.Expr:
        """Lazy counterpart of :meth:`_prepare_y` as a polars expression."""
        return pl.col(target_col).cast(pl.Float32)

    def _resolve_n_outputs(self) -> int:
        return 1

    def _make_loss(self) -> Loss:
        if isinstance(self.loss, Loss):
            return copy.deepcopy(self.loss)
        loss_spec = ModuleParserSpec(self.loss or self._default_loss, allowed=LOSSES)
        return make_loss(loss_spec.module_name(), **loss_spec.kwargs())

    def _fit_target_meta(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        y: YLike,
        target_col: str | None,
    ) -> None:
        """Fit task-specific target metadata, such as classifier classes."""

    def _build_model(
        self,
        *,
        cards: list[int],
        embedding_dims: list[int],
        ple_bins: list[torch.Tensor] | None,
        multihash_n_inputs: int,
        embedding_encoders: dict[str, str | torch.nn.Module] | None,
        embedding_input_dims: dict[str, int],
        n_outputs: int,
        use_coral_head: bool,
    ) -> torch.nn.Module:
        if use_coral_head or n_outputs != 1:
            raise ValueError("FinalMLP currently supports scalar outputs only")
        multihash_config = multihash_encoder_config(self.multihash_encoder)
        return build_final_mlp(
            num_feature_names=self.preprocessor_.num_cols_,
            cat_feature_names=self.preprocessor_.cat_cols_,
            cardinalities=cards,
            embedding_dims=embedding_dims,
            mlp1_hidden_units=self.mlp1_hidden_units,
            mlp1_hidden_activations=self.mlp1_hidden_activations,
            mlp1_dropout=self.mlp1_dropout,
            mlp1_batch_norm=self.mlp1_batch_norm,
            mlp2_hidden_units=self.mlp2_hidden_units,
            mlp2_hidden_activations=self.mlp2_hidden_activations,
            mlp2_dropout=self.mlp2_dropout,
            mlp2_batch_norm=self.mlp2_batch_norm,
            use_fs=self.use_fs,
            fs_hidden_units=self.fs_hidden_units,
            fs1_context=self.fs1_context,
            fs2_context=self.fs2_context,
            num_heads=self.num_heads,
            context_dim=self.embedding_dim or 10,
            use_pytorch_init=self.use_pytorch_init,
            num_encoder=self.num_encoder,
            cat_encoder=self.cat_encoder,
            num_encoder_bins=ple_bins,
            multihash_feature_names=self.preprocessor_.multihash_cols_,
            multihash_encoder=self.multihash_encoder,
            multihash_n_hashes=multihash_config["n_hashes"],
            embedding_encoders=embedding_encoders,
            embedding_input_dims=embedding_input_dims or None,
        )

    def fit(
        self,
        X: XLike,
        y: YLike = None,
        group: GroupLike = None,
        eval_set: EvalSet | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the estimator on ``X`` and ``y``.

        Parameters
        ----------
        X : numpy.ndarray, pandas.DataFrame, polars.DataFrame, or polars.LazyFrame
            Training features. String categoricals and NaNs are handled natively.
            With a lazy frame, the data is preprocessed and streamed through a
            temporary Arrow IPC file.
        y : array-like or str, default=None
            Target values, or a target-column name when ``X`` is a polars frame.
        group : array-like or str or None, default=None
            Per-row query ids for ranking losses. With a lazy frame, pass the
            group-column name. Group-aware losses require this argument.
        eval_set : tuple or None, default=None
            Validation data as ``(X_val, y_val)`` or ``(X_val, y_val, group_val)``.
            Required when ``early_stopping_rounds`` is set.
        **kwargs
            Unsupported. Passing fit parameters such as ``sample_weight`` raises
            :class:`TypeError`.

        Returns
        -------
        self
            The fitted estimator.

        """
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.fit() got unexpected keyword "
                f"argument(s): {sorted(kwargs)!r}",
            )
        if self.eval_metric is not None and not callable(self.eval_metric):
            raise TypeError(
                "eval_metric must be a callable metric_fn(y_true, y_pred) -> float "
                "(e.g. sklearn.metrics.roc_auc_score) or None; "
                f"got {self.eval_metric!r}",
            )
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self._rng = np.random.default_rng(self.random_state)

        frame = to_polars(X)
        is_lazy = isinstance(frame, pl.LazyFrame)
        target_col = y if isinstance(y, str) else None
        group_col = group if isinstance(group, str) else None
        if is_lazy and target_col is None:
            raise ValueError("With a polars LazyFrame, `y` must be a column name in X.")
        if is_lazy and group is not None and group_col is None:
            raise ValueError("With a polars LazyFrame, `group` must be a column name in X.")
        if not is_lazy:
            validate_X(X if isinstance(X, np.ndarray) else frame)
        if not isinstance(y, str):
            y = validate_y(y)

        exclude = tuple(c for c in (target_col, group_col) if c is not None)
        multihash_config = multihash_encoder_config(self.multihash_encoder)
        self.preprocessor_ = TabularPreprocessor(
            num_features=list(self.num_features) if self.num_features is not None else None,
            cat_features=list(self.cat_features) if self.cat_features is not None else None,
            normalize=self.normalize_numeric,
            n_quantiles=self.n_quantiles,
            multihash_features=(
                list(self.multihash_features) if self.multihash_features is not None else None
            ),
            multihash_cardinality=multihash_config["cardinality"],
            multihash_n_hashes=multihash_config["n_hashes"],
            embedding_features=(
                dict(self.embedding_features) if self.embedding_features is not None else None
            ),
            numeric_nan_fill=self.numeric_nan_fill,
        ).fit(frame, exclude=exclude)

        self._fit_target_meta(frame, y, target_col)

        ple_bins: list[torch.Tensor] | None = None
        if (
            isinstance(self.num_encoder, str)
            and self.num_encoder.split(":", 1)[0].lower() == "ple"
            and self.preprocessor_.num_cols_
        ):
            ple_bins = self.preprocessor_.fit_ple_bins(frame, n_bins=self.ple_n_bins)

        cards = self.preprocessor_.cardinalities_
        embedding_dims = [
            self.embedding_dim
            if self.embedding_dim is not None
            else min(32, max(2, round(1.6 * cardinality**0.56)))
            for cardinality in cards
        ]
        loss_fn = self._make_loss()
        is_coral = isinstance(loss_fn, CORALLayerLoss)
        if is_coral and not hasattr(self, "n_classes_"):
            estimator_name = type(self).__name__
            model_name = estimator_name.removesuffix("Regressor").removesuffix("Ranker")
            raise ValueError(
                f"loss='coral_layer' is only supported by {model_name}Classifier",
            )
        n_outputs = int(self.n_classes_) if is_coral else self._resolve_n_outputs()
        embedding_input_dims = self.preprocessor_.embedding_input_dims_
        if self.embedding_encoders and not embedding_input_dims:
            raise ValueError("embedding_encoders requires embedding_features")
        embedding_encoders = None
        if embedding_input_dims:
            embedding_encoders = dict.fromkeys(embedding_input_dims, "tower") | dict(
                self.embedding_encoders or {},
            )

        model = self._build_model(
            cards=cards,
            embedding_dims=embedding_dims,
            ple_bins=ple_bins,
            multihash_n_inputs=self.preprocessor_.multihash_n_inputs_,
            embedding_encoders=embedding_encoders,
            embedding_input_dims=embedding_input_dims,
            n_outputs=n_outputs,
            use_coral_head=is_coral,
        )

        with contextlib.ExitStack() as cleanup:
            router = DataRouter(
                self.preprocessor_,
                batch_size=self.batch_size,
                chunk_rows=self.chunk_rows,
                rng=self._rng,
                encode_target=self._prepare_y,
                target_expr=self._y_expr,
            )
            train_source = router.build_train_source(frame, y, group, cleanup=cleanup)
            val_source = router.build_eval_source(
                eval_set,
                group_col=group_col,
                cleanup=cleanup,
                require_group=self.eval_metric is not None and self.eval_metric_group_aware,
            )
            if self.early_stopping_rounds is not None and val_source is None:
                raise ValueError("early_stopping_rounds requires eval_set")

            train_out = TrainingRun(
                model,
                loss_fn,
                train_source,
                val_source,
                lr=self.lr,
                weight_decay=self.weight_decay,
                optimizer=OptimizerConfig(
                    optimizer_type=self.optimizer,
                    **(self.optimizer_kwargs or {}),
                ),
                epochs=self.epochs,
                accelerator_config=self.accelerator_config,
                training_options=self.training_options,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=self.verbose,
                eval_metric_fn=self.eval_metric,
                eval_metric_name=self.eval_metric_name,
                eval_metric_direction=self.eval_metric_direction,
                eval_metric_group_aware=self.eval_metric_group_aware,
                lr_scheduler=build_lr_scheduler_config(self.lr_scheduler),
                grad_clip_norm=self.grad_clip_norm,
                embedding_regularizer=self.embedding_regularizer,
                ema_decay=self.ema_decay,
            ).run()
            module, history = train_out.module(), train_out.metrics()["history"]

        self.model_ = module.model().cpu()
        self.loss_ = module.loss_fn().cpu()
        self.history_ = history
        self.n_features_in_ = (
            len(self.preprocessor_.num_cols_)
            + len(self.preprocessor_.cat_cols_)
            + len(self.preprocessor_.multihash_cols_)
            + len(self.preprocessor_.embedding_cols_)
        )
        self.feature_names_in_ = np.asarray(
            self.preprocessor_.num_cols_
            + self.preprocessor_.cat_cols_
            + self.preprocessor_.multihash_cols_
            + list(self.preprocessor_.embedding_cols_.values()),
        )
        return self

    def save(self, path: str | Path) -> None:
        """Pickle the fitted estimator to ``path``."""
        with Path(path).open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load an estimator saved with :meth:`save`."""
        with Path(path).open("rb") as file:
            obj = pickle.load(file)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected saved {cls.__name__}, got {type(obj).__name__}")
        return obj

    def _decision_scores(self, X: XLike) -> np.ndarray:
        check_is_fitted(self, "model_")
        if not isinstance(X, pl.LazyFrame):
            validate_X(
                X,
                expected_features=self.n_features_in_,
                estimator_name=type(self).__name__,
            )
        return score_tabular_model(
            model=self.model_,
            preprocessor=self.preprocessor_,
            frame=to_polars(X),
            batch_size=self.batch_size,
            chunk_rows=self.chunk_rows,
            accelerator_config=self.accelerator_config,
        )


class FinalMLPClassifier(ClassifierMixin, FinalMLPBase):
    """FinalMLP binary classifier.

    Targets may be boolean, integer, or string labels. This estimator currently
    supports exactly two classes and uses ``bce`` by default. Fitted labels are
    exposed through ``classes_``. See :class:`FinalMLPBase` for parameters.

    Examples
    --------
    >>> from scikit_rank import FinalMLPClassifier
    >>> classifier = FinalMLPClassifier(epochs=5, num_features=["num"])
    >>> classifier.fit(X, y)  # doctest: +SKIP
    >>> classifier.predict_proba(X).shape  # doctest: +SKIP
    (n_samples, 2)

    """

    _default_loss = "bce"

    def _fit_target_meta(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        y: YLike,
        target_col: str | None,
    ) -> None:
        self._classification_target_ = ClassificationTarget.fit(frame, y, target_col)
        self._label_encoder_ = self._classification_target_.label_encoder
        self.classes_ = self._classification_target_.classes
        self.n_classes_ = self._classification_target_.n_classes

    def _resolve_n_outputs(self) -> int:
        return 1 if self.n_classes_ <= 2 else self.n_classes_

    def fit(
        self,
        X: XLike,
        y: YLike = None,
        group: GroupLike = None,
        eval_set: EvalSet | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the classifier after inferring and encoding its classes."""
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.fit() got unexpected keyword "
                f"argument(s): {sorted(kwargs)!r}",
            )
        if not isinstance(X, pl.LazyFrame):
            validate_X(X if isinstance(X, np.ndarray) else to_polars(X))
        if not isinstance(y, str):
            y = validate_y(y)
        frame = to_polars(X)
        target_col = y if isinstance(y, str) else None
        self._fit_target_meta(frame, y, target_col)
        self._default_loss = "bce" if self.n_classes_ <= 2 else "cross_entropy"
        self._validate_classifier_target()
        return super().fit(X, y=y, group=group, eval_set=eval_set)

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        return self._classification_target_.encode(y)

    def _y_expr(self, target_col: str) -> pl.Expr:
        return self._classification_target_.expression(target_col)

    def _validate_classifier_target(self) -> None:
        if self.n_classes_ != 2:
            raise ValueError(
                "FinalMLPClassifier supports binary classification only; "
                f"got {self.n_classes_} classes",
            )
        if isinstance(self._make_loss(), (CrossEntropyLoss, CORALLayerLoss)):
            raise TypeError(
                "FinalMLPClassifier currently requires a scalar-output loss; "
                "cross_entropy and coral_layer are not supported",
            )

    def predict_proba(self, X: XLike) -> np.ndarray:
        """Predict class probabilities for ``X``."""
        return class_probabilities(self._decision_scores(X))

    def predict(self, X: XLike) -> np.ndarray:
        """Predict class labels for ``X``."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class FinalMLPRegressor(RegressorMixin, FinalMLPBase):
    """FinalMLP regressor with an ``mse`` loss by default.

    See :class:`FinalMLPBase` for parameters.

    Examples
    --------
    >>> from scikit_rank import FinalMLPRegressor
    >>> regressor = FinalMLPRegressor(epochs=5, num_features=["num"])
    >>> predictions = regressor.fit(X, y_reg).predict(X)  # doctest: +SKIP

    """

    _default_loss = "mse"

    def predict(self, X: XLike) -> np.ndarray:
        """Predict continuous targets for ``X``."""
        return self._decision_scores(X)


class FinalMLPRanker(FinalMLPBase):
    """FinalMLP learning-to-rank estimator.

    The default ``lambdarank`` loss, and other group-aware losses, require one
    query id per row through ``fit(X, y, group=...)``. Ranking batches preserve
    complete queries and :meth:`predict` returns scores where larger is better.
    See :class:`FinalMLPBase` for parameters.

    Examples
    --------
    >>> from scikit_rank import FinalMLPRanker
    >>> ranker = FinalMLPRanker(loss="listwise", epochs=5)
    >>> scores = ranker.fit(df, y="click", group="impression_id").predict(df)  # doctest: +SKIP

    """

    _default_loss = "lambdarank"

    def fit(
        self,
        X: XLike,
        y: YLike = None,
        group: GroupLike = None,
        eval_set: EvalSet | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fit the ranker, requiring groups for group-aware losses."""
        loss_fn = self._make_loss()
        if group is None and loss_fn.requires_group:
            raise ValueError(
                "FinalMLPRanker.fit requires `group` (per-row query ids or a column name).",
            )
        return super().fit(X, y=y, group=group, eval_set=eval_set, **kwargs)

    def predict(self, X: XLike) -> np.ndarray:
        """Predict ranking scores for ``X``."""
        scores = self._decision_scores(X)
        return scores.sum(axis=1) if scores.ndim == 2 else scores

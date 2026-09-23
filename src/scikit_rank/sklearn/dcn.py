"""sklearn-compatible estimators: DCNClassifier, DCNRegressor, DCNRanker.

API follows the LightGBM/XGBoost sklearn wrappers: flat ``__init__``
hyperparameters (sklearn ``get_params``/``set_params``/``clone`` work out of
the box), ``fit(X, y, group=..., eval_set=...)``, ``predict``,
``predict_proba`` where applicable.

Accepted inputs: numpy.ndarray, pandas.DataFrame, polars.DataFrame and
polars.LazyFrame (lazy training streams preprocessed data through a temp
Arrow IPC file). All batch-source construction is delegated to
:class:`scikit_rank.sklearn._data_router.DataRouter`; see ``data.py`` for the
sources themselves. When X is a polars (Lazy)Frame, ``y`` and ``group`` may be
given as column names.
"""

from __future__ import annotations
import contextlib
import copy
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
import torch
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from scikit_rank.data import to_polars
from scikit_rank.factories import build_dcnv2, build_lr_scheduler_config, multihash_encoder_config
from scikit_rank.modules.dcn import CoralLayer
from scikit_rank.modules.losses import LOSSES, CORALLayerLoss, Loss, make_loss
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


class DCNBase(BaseEstimator):
    """Shared fit/predict machinery for the DCNv2 estimators.

    Not used directly -- instantiate :class:`DCNClassifier`, :class:`DCNRegressor`,
    or :class:`DCNRanker`. This base holds every hyperparameter and the common
    training/inference loop; the subclasses only add task-specific target handling
    and prediction. All estimators follow the LightGBM/XGBoost sklearn-wrapper
    convention: every hyperparameter is an explicit ``__init__`` keyword, so
    :func:`sklearn.base.clone`, ``get_params``/``set_params``, and
    ``GridSearchCV`` work out of the box.

    Parameters
    ----------
    hidden_units : list[int], default=(256, 128)
        Layer widths of the deep (MLP) branch.
    cross_layers : int, default=3
        Number of DCNv2 feature-crossing layers.
    cross_rank : int or None, default=None
        Rank of the low-rank factorization of each cross weight matrix. ``None``
        uses the full-rank cross weights.
    embedding_dim : int or None, default=None
        Embedding size for every categorical feature. ``None`` picks a per-column
        size with the fast.ai heuristic ``min(32, max(2, round(1.6 * card**0.56)))``.
    dropout : float, default=0.0
        Dropout probability applied in the deep branch.
    structure : {"stacked", "parallel"}, default="stacked"
        How the cross and deep branches are composed. ``"stacked"`` feeds the
        cross output into the deep branch; ``"parallel"`` concatenates them.
    num_encoder : str or torch.nn.Module, default="identity"
        Numeric-feature encoder spec, e.g. ``"identity"`` or ``"ple"`` (piecewise
        linear encoding, whose bins are fit from training quantiles). A custom
        ``Module`` is used as-is.
    cat_encoder : str or torch.nn.Module, default="per_feature"
        Categorical-feature encoder spec (e.g. one embedding table per feature).
    gated_cross : bool, default=False
        Enable the gated variant of the cross layers.
    cross_type : str, default="standard"
        Cross-layer variant selector.
    mask_ratio : float, default=0.5
        Masking ratio used by mask-enabled cross variants.
    activation : str, default="relu"
        Activation function name for the deep branch.
    batch_norm : bool, default=False
        Apply batch normalization in the deep branch.
    use_moe : bool, default=False
        Replace the deep branch with a mixture-of-experts block.
    num_experts : int, default=4
        Number of experts when ``use_moe=True``.
    moe_top_k : int, default=2
        Number of experts routed per row when ``use_moe=True``.
    use_inner_cross_layers : bool, default=False
        Enable the inner cross-layer variant.
    use_pytorch_init : bool, default=False
        Preserve constructor initialization instead of applying the common
        scikit-rank initializer.
    loss : str or Loss or None, default=None
        Loss spec string (e.g. ``"bce"``, ``"bpr:sampling=all_pairs"``,
        ``"lambdarank"``, ``"cross_entropy"``, ``"coral_layer"``) or a :class:`Loss`
        instance. ``None`` uses the subclass default (``bce``/``cross_entropy`` for
        classification, ``mse`` for regression, ``lambdarank`` for ranking).
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=0.0
        Weight-decay (L2) coefficient passed to the optimizer.
    optimizer : str, default="adamw"
        Optimizer name.
    optimizer_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to the optimizer.
    epochs : int, default=10
        Maximum number of training epochs.
    batch_size : int, default=1024
        Mini-batch size. For ranking, batches never split a group.
    early_stopping_rounds : int or None, default=None
        Stop after this many epochs without eval-metric improvement. Requires
        ``eval_set`` to be passed to :meth:`fit`.
    eval_metric : callable or None, default=None
        Validation metric ``metric_fn(y_true, y_pred[, group]) -> float`` (e.g.
        :func:`sklearn.metrics.roc_auc_score`). ``None`` monitors the eval loss.
    eval_metric_name : str, default="metric"
        Name the metric is logged under in ``history_``.
    eval_metric_direction : {"max", "min"}, default="max"
        Whether a higher or lower ``eval_metric`` value is better (drives model
        selection and early stopping).
    eval_metric_group_aware : bool, default=False
        If ``True``, ``eval_metric`` is called as ``metric_fn(y_true, y_pred, group)``
        with per-group ids (for ranking metrics such as NDCG).
    num_features : sequence of str or None, default=None
        Explicit numeric columns. ``None`` infers them from dtypes.
    cat_features : sequence of str or None, default=None
        Explicit categorical columns. ``None`` infers them from dtypes.
    multihash_features : sequence of str or None, default=None
        Columns routed through a shared hashed (Unified Embedding) table -- useful
        for very high-cardinality ids.
    multihash_encoder : str or torch.nn.Module, default="multihash"
        Encoder spec for ``multihash_features`` (controls cardinality / hash count).
    embedding_features : dict[str, str] or None, default=None
        Named external embedding streams, mapping stream name to the column that
        holds a precomputed embedding vector per row.
    embedding_encoders : dict[str, str | torch.nn.Module] or None, default=None
        Per-stream encoder for ``embedding_features`` (defaults to a ``"tower"``).
    normalize_numeric : bool or str or None, default=True
        Numeric normalization strategy (e.g. quantile normalization) applied by
        the preprocessor.
    n_quantiles : int, default=1000
        Number of quantiles for quantile normalization.
    numeric_nan_fill : {"median", "zero"}, default="median"
        How missing numeric values are imputed (statistics fit on train only).
    ple_n_bins : int, default=42
        Number of bins for the PLE numeric encoder when ``num_encoder="ple"``.
    lr_scheduler : str or None, default=None
        LR-scheduler spec, e.g. ``"plateau:patience=0;factor=0.1;min_lr=1e-6"``.
    grad_clip_norm : float or None, default=None
        Global gradient-norm clip value. ``None`` disables clipping.
    embedding_regularizer : float, default=0.0
        Coefficient of the coupled embedding L2 penalty added to the train loss.
    ema_decay : float or None, default=None
        If set (in ``(0, 1)``), keep an exponential moving average of the weights
        and use it for evaluation / final model.
    chunk_rows : int, default=100_000
        Arrow streaming chunk size for the lazy (``polars.LazyFrame``) path and
        for chunked inference.
    random_state : int or None, default=None
        Seed for torch and numpy RNGs. Training is not bit-for-bit deterministic
        on GPU even with a fixed seed.
    accelerator_config : dict or None, default=None
        Options forwarded to 🤗 Accelerate (e.g. ``{"cpu": True}``, mixed precision,
        DDP). Also selects the inference device.
    verbose : bool, default=False
        Print a training progress bar and per-epoch logs.

    Attributes
    ----------
    model_ : torch.nn.Module
        The fitted DCNv2 network (kept on CPU for stable pickling).
    loss_ : Loss
        The instantiated loss module.
    history_ : list[dict[str, float]]
        Per-epoch records with ``train_loss`` and, when ``eval_set`` is given,
        ``val_loss`` / ``val_<eval_metric_name>``.
    preprocessor_ : TabularPreprocessor
        The fitted feature preprocessor.
    n_features_in_ : int
        Number of input features seen during :meth:`fit`.
    feature_names_in_ : numpy.ndarray
        Names of the input features, in preprocessing order.

    """

    _default_loss = "bce"

    def __init__(  # noqa: PLR0913 -- sklearn estimator: every hyperparam is an explicit kwarg
        self,
        *,
        hidden_units: list[int] | tuple[int, ...] = (256, 128),
        cross_layers: int = 3,
        cross_rank: int | None = None,
        embedding_dim: int | None = None,
        dropout: float = 0.0,
        structure: Literal["stacked", "parallel"] = "stacked",
        num_encoder: str | torch.nn.Module = "identity",
        cat_encoder: str | torch.nn.Module = "per_feature",
        gated_cross: bool = False,
        cross_type: str = "standard",
        mask_ratio: float = 0.5,
        activation: str = "relu",
        batch_norm: bool = False,
        use_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 2,
        use_inner_cross_layers: bool = False,
        use_pytorch_init: bool = False,
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
        self.hidden_units = hidden_units
        self.cross_layers = cross_layers
        self.cross_rank = cross_rank
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.structure = structure
        self.num_encoder = num_encoder
        self.gated_cross = gated_cross
        self.cross_type = cross_type
        self.mask_ratio = mask_ratio
        self.activation = activation
        self.batch_norm = batch_norm
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.use_inner_cross_layers = use_inner_cross_layers
        self.use_pytorch_init = use_pytorch_init
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
        self.accelerator_config = accelerator_config
        self.training_options = training_options
        self.chunk_rows = chunk_rows
        self.random_state = random_state
        self.verbose = verbose

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
            loss_fn = copy.deepcopy(self.loss)
        else:
            loss_spec = ModuleParserSpec(
                self.loss or self._default_loss,
                allowed=LOSSES,
            )
            loss_fn = make_loss(loss_spec.module_name(), **loss_spec.kwargs())
        return loss_fn

    def _fit_target_meta(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        y: YLike,
        target_col: str | None,
    ) -> None:
        """Fit subclass-specific target metadata (e.g. classes_)."""

    def fit(
        self,
        X: XLike,
        y: YLike = None,
        group: GroupLike = None,
        eval_set: EvalSet | None = None,
        **kwargs: Any,
    ) -> DCNBase:
        """Fit the estimator on ``X`` and ``y``.

        Parameters
        ----------
        X : numpy.ndarray, pandas.DataFrame, polars.DataFrame, or polars.LazyFrame
            Training features. String categoricals and NaNs are handled natively.
            A ``LazyFrame`` is preprocessed and streamed through a temporary Arrow
            IPC file (out-of-core training); with a ``LazyFrame`` the target and
            group must be given as column names.
        y : array-like or str, default=None
            Target values, or -- when ``X`` is a polars (Lazy)Frame -- the name of
            the target column in ``X``.
        group : array-like or str or None, default=None
            Per-row query/group ids for ranking (array or column name). Ignored by
            the classifier/regressor; required by :class:`DCNRanker` for
            group-aware losses.
        eval_set : tuple or None, default=None
            Validation data as ``(X_val, y_val)`` or ``(X_val, y_val, group_val)``.
            Enables ``val_loss``/eval-metric logging and is required when
            ``early_stopping_rounds`` is set.
        **kwargs
            Not accepted; passing any (e.g. ``sample_weight``) raises ``TypeError``.

        Returns
        -------
        self : DCNBase
            The fitted estimator.

        """
        if kwargs:
            # Reject silently-swallowed fit kwargs (notably sklearn's
            # `sample_weight`, which is not supported by this estimator).
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
            raise ValueError(
                "With a polars LazyFrame, `group` must be a column name in X.",
            )
        # Input validation (sklearn-compliance). For lazy frames and column-name
        # ``y`` we skip array-level y validation -- the polars cast pipeline
        # raises informatively if the column doesn't exist or is malformed.
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

        # data-driven PLE bins from training quantiles (only when the spec is
        # the string 'ple'; a user-supplied Module is left untouched)
        ple_bins: list[torch.Tensor] | None = None
        if (
            isinstance(self.num_encoder, str)
            and self.num_encoder.split(":", 1)[0].lower() == "ple"
            and self.preprocessor_.num_cols_
        ):
            ple_bins = self.preprocessor_.fit_ple_bins(frame, n_bins=self.ple_n_bins)

        # model
        cards = self.preprocessor_.cardinalities_
        mh_n_inputs = self.preprocessor_.multihash_n_inputs_
        emb_dims = [
            self.embedding_dim
            if self.embedding_dim is not None
            else min(
                32,
                max(2, round(1.6 * c**0.56)),
            )  # fast-ai logic to pick embedding_dim
            for c in cards
        ]
        loss_fn = self._make_loss()
        is_coral = isinstance(loss_fn, CORALLayerLoss)
        if is_coral and not hasattr(self, "n_classes_"):
            raise ValueError("loss='coral_layer' is only supported by DCNClassifier")
        # CORAL needs num_classes (K) instead of out_features (K-1) at the head;
        # the factory interprets n_outputs accordingly when use_coral_head=True.
        n_outputs = int(self.n_classes_) if is_coral else self._resolve_n_outputs()
        embedding_input_dims = self.preprocessor_.embedding_input_dims_
        if self.embedding_encoders and not embedding_input_dims:
            raise ValueError("embedding_encoders requires embedding_features")
        embedding_encoders = None
        if embedding_input_dims:
            embedding_encoders = dict.fromkeys(embedding_input_dims, "tower") | dict(
                self.embedding_encoders or {},
            )

        model = build_dcnv2(
            n_num_features=len(self.preprocessor_.num_cols_),
            cardinalities=cards,
            embedding_dims=emb_dims,
            cross_layers=self.cross_layers,
            cross_rank=self.cross_rank,
            hidden_units=list(self.hidden_units),
            dropout=self.dropout,
            structure=self.structure,
            num_encoder=self.num_encoder,
            cat_encoder=self.cat_encoder,
            gated_cross=self.gated_cross,
            cross_type=self.cross_type,
            mask_ratio=self.mask_ratio,
            activation=self.activation,
            batch_norm=self.batch_norm,
            use_moe=self.use_moe,
            num_experts=self.num_experts,
            moe_top_k=self.moe_top_k,
            use_inner_cross_layers=self.use_inner_cross_layers,
            use_pytorch_init=self.use_pytorch_init,
            num_encoder_bins=ple_bins,
            multihash_encoder=self.multihash_encoder,
            multihash_n_inputs=mh_n_inputs or None,
            embedding_encoders=embedding_encoders,
            embedding_input_dims=embedding_input_dims or None,
            n_outputs=n_outputs,
            use_coral_head=is_coral,
        )

        # data sources; temp Arrow files (lazy path) are cleaned up on exit
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

            lr_scheduler_config = build_lr_scheduler_config(self.lr_scheduler)
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
                lr_scheduler=lr_scheduler_config,
                grad_clip_norm=self.grad_clip_norm,
                embedding_regularizer=self.embedding_regularizer,
                ema_decay=self.ema_decay,
            ).run()
            module, history = train_out.module(), train_out.metrics()["history"]

        # keep fitted modules on CPU: pickling and re-fitting stay trivial
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
        """Pickle the fitted estimator to ``path``.

        Models are kept on CPU after ``fit``/``predict``, so plain pickle is
        stable across CPU/GPU machines.
        """
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> DCNBase:
        """Load an estimator saved with :meth:`save`."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)  # noqa: S301
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


class DCNClassifier(ClassifierMixin, DCNBase):
    """DCNv2 classifier (binary or multiclass).

    A scikit-learn ``ClassifierMixin``. Binary problems train a single logit with
    ``bce`` (the default; any pointwise/ordinal loss also works). Multiclass
    problems train K logits with ``cross_entropy``. Targets may be integers,
    strings, or booleans; the fitted classes are stored in ``classes_`` and
    predictions are mapped back to the original labels.

    See :class:`DCNBase` for the full list of hyperparameters.

    Attributes
    ----------
    classes_ : numpy.ndarray
        The unique class labels seen during :meth:`fit`.
    n_classes_ : int
        Number of classes.

    Examples
    --------
    >>> import polars as pl
    >>> from scikit_rank import DCNClassifier
    >>> X = pl.DataFrame({"num": [0.1, 1.2, -0.3], "cat": ["a", "b", "a"]})
    >>> clf = DCNClassifier(epochs=5, num_features=["num"], cat_features=["cat"])
    >>> clf.fit(X, [0, 1, 0])                       # doctest: +SKIP
    >>> clf.predict_proba(X).shape                  # doctest: +SKIP
    (3, 2)

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
    ) -> DCNClassifier:
        """Fit the classifier, inferring ``classes_`` before training.

        Same signature as :meth:`DCNBase.fit`. ``y`` (or the column it names) may
        hold integer, string, or boolean labels; two classes train a single-logit
        ``bce`` head and more than two train a ``cross_entropy`` head.
        """
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.fit() got unexpected keyword "
                f"argument(s): {sorted(kwargs)!r}",
            )
        # Validate inputs early so target-meta probing sees a clean y. Lazy
        # frames and column-name y skip array-level validation (handled later).
        is_lazy_X = isinstance(X, pl.LazyFrame)
        if not is_lazy_X:
            validate_X(X if isinstance(X, np.ndarray) else to_polars(X))
        if not isinstance(y, str):
            y = validate_y(y)
        frame = to_polars(X)
        target_col = y if isinstance(y, str) else None
        self._fit_target_meta(frame, y, target_col)
        self._default_loss = "bce" if self.n_classes_ <= 2 else "cross_entropy"
        return super().fit(X, y=y, group=group, eval_set=eval_set)

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        return self._classification_target_.encode(y)

    def _y_expr(self, target_col: str) -> pl.Expr:
        return self._classification_target_.expression(target_col)

    def predict_proba(self, X: XLike) -> np.ndarray:
        """Predict class probabilities for ``X``.

        Parameters
        ----------
        X : array-like, DataFrame, or LazyFrame
            Samples to score, with the same features seen during :meth:`fit`.

        Returns
        -------
        proba : numpy.ndarray of shape (n_samples, n_classes)
            Per-class probabilities. Columns are ordered as ``classes_`` and each
            row sums to 1. Binary models return two columns ``[P(neg), P(pos)]``.

        """
        scores = self._decision_scores(X)
        if isinstance(self.model_.head(), CoralLayer) and scores.ndim > 1:
            # CORAL logits are cumulative P(Y >= k), k=1..K-1. Convert to
            # mutually-exclusive class probabilities and enforce monotonicity
            # defensively in case learned raw biases cross.
            p_ge = np.minimum.accumulate(expit(scores), axis=1)
            return np.concatenate(
                [
                    1.0 - p_ge[:, :1],
                    p_ge[:, :-1] - p_ge[:, 1:],
                    p_ge[:, -1:],
                ],
                axis=1,
            )
        return class_probabilities(scores)

    def predict(self, X: XLike) -> np.ndarray:
        """Predict class labels for ``X``.

        Returns
        -------
        labels : numpy.ndarray of shape (n_samples,)
            The predicted label (from ``classes_``) with the highest probability.

        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class DCNRegressor(RegressorMixin, DCNBase):
    """DCNv2 regressor.

    A scikit-learn ``RegressorMixin`` predicting a single continuous target,
    trained with ``mse`` by default. See :class:`DCNBase` for the full list of
    hyperparameters.

    Examples
    --------
    >>> from scikit_rank import DCNRegressor
    >>> reg = DCNRegressor(epochs=5, num_features=["num"], cat_features=["cat"])
    >>> reg.fit(X, y_reg)          # doctest: +SKIP
    >>> reg.predict(X).shape       # doctest: +SKIP
    (n_samples,)

    """

    _default_loss = "mse"

    def predict(self, X: XLike) -> np.ndarray:
        """Predict continuous targets for ``X``.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            The predicted target values.

        """
        return self._decision_scores(X)


class DCNRanker(DCNBase):
    """DCNv2 learning-to-rank estimator.

    Trains a per-row relevance score with a ranking loss (``lambdarank`` by
    default; also ``bpr``, listwise/softmax, etc.). Call
    ``fit(X, y, group=...)`` where ``group`` is a per-row query id array, or a
    column name when ``X`` is a polars (Lazy)Frame. Batches never split a group,
    so pairwise/listwise losses always see complete groups. See :class:`DCNBase`
    for the full list of hyperparameters.

    Examples
    --------
    >>> from scikit_rank import DCNRanker
    >>> ranker = DCNRanker(loss="listwise", epochs=5)
    >>> ranker.fit(df, y="click", group="impression_id")   # doctest: +SKIP
    >>> scores = ranker.predict(df)                        # doctest: +SKIP

    """

    _default_loss = "lambdarank"

    def fit(
        self,
        X: XLike,
        y: YLike = None,
        group: GroupLike = None,
        eval_set: EvalSet | None = None,
        **kwargs: Any,
    ) -> DCNRanker:
        """Fit the ranker on grouped data.

        Same signature as :meth:`DCNBase.fit`. ``group`` (a per-row query-id
        array, or a column name when ``X`` is a polars (Lazy)Frame) is required
        for group-aware losses and raises ``ValueError`` if missing.
        """
        loss_fn = self._make_loss()
        if group is None and loss_fn.requires_group:
            raise ValueError(
                "DCNRanker.fit requires `group` (per-row query ids or a column name).",
            )
        return super().fit(X, y=y, group=group, eval_set=eval_set, **kwargs)

    def predict(self, X: XLike) -> np.ndarray:
        """Ranking scores (higher = more relevant)."""
        scores = self._decision_scores(X)
        if scores.ndim == 2:  # e.g. coral_layer head: sum K-1 logits to a scalar score
            scores = scores.sum(axis=1)
        return scores

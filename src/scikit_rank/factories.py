"""Factories that compose model backbones from user-facing hyperparameters.

This module resolves string encoder specs, sizes heads, and wires feature
streams into DCNv2, FinalMLP, DESTINE, and TabM. Callers (typically
:mod:`scikit_rank.sklearn`) supply scalars and string specs and receive a ready,
parameter-initialized model.
"""

from __future__ import annotations
import copy
import math
from typing import TYPE_CHECKING, Literal

import tabm as tabm_lib
import torch
from einops.layers.torch import EinMix

from scikit_rank.modules.dcn import (
    CategoricalEmbeddings,
    CoralLayer,
    CrossLayer,
    CrossNetwork,
    DCNv2,
    DeepNetwork,
    EmbeddingTower,
    LinearNumericEncoder,
    MultiHashEmbeddings,
    NumericEncoder,
    ParallelCrossDeep,
    PiecewiseLinearEncoder,
    PLREncoder,
    StackedCrossDeep,
    UnifiedEmbeddings,
)
from scikit_rank.modules.destine import (
    DESTINE,
    DenseFieldEncoder,
    DESTINEWide,
    DisentangledSelfAttention,
    MultiHashFieldEncoder,
    NumericFieldEmbeddings,
    ReshapedFieldEncoder,
)
from scikit_rank.modules.final_mlp import (
    FeatureSelection,
    FinalMLP,
    InputSlice,
    InteractionAggregation,
)
from scikit_rank.modules.finalnet import FinalBlock, FinalNet, FinalNetFieldGate
from scikit_rank.modules.reducers import Concat
from scikit_rank.modules.tabm import CoralEnsemble, EnsembleAggregation, TabM
from scikit_rank.train.optimizers import LRSchedulerConfig
from scikit_rank.utils import ModuleParserSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

_PLE_DEFAULTS = {
    "embedding_dim": None,
    "activation": False,
    "feature_dropout": 0.0,
}

_PLR_DEFAULTS = {
    "n_freq": 32,
    "sigma": 1.0,
    "embedding_dim": 16,
    "activation": "relu",
    "feature_dropout": 0.0,
}

_LINEAR_DEFAULTS = {
    "embedding_dim": 16,
}

_UNIFIED_EMB_DEFAULTS = {
    "embedding_dim": 16,
}

_NUM_REGISTRY: dict[str, tuple[type[torch.nn.Module], dict]] = {
    "identity": (NumericEncoder, {}),
    "ple": (PiecewiseLinearEncoder, _PLE_DEFAULTS),
    "plr": (PLREncoder, _PLR_DEFAULTS),
    "linear": (LinearNumericEncoder, _LINEAR_DEFAULTS),
}

_CAT_REGISTRY: dict[str, tuple[type[torch.nn.Module], dict]] = {
    "per_feature": (CategoricalEmbeddings, {}),
    "unified": (UnifiedEmbeddings, _UNIFIED_EMB_DEFAULTS),
}

_MULTIHASH_DEFAULTS = {
    "cardinality": 100_000,
    "n_hashes": 2,
    "embedding_dim": 16,
}

_MULTIHASH_REGISTRY: dict[str, tuple[type[torch.nn.Module], dict]] = {
    "multihash": (MultiHashEmbeddings, _MULTIHASH_DEFAULTS),
}

_EMBEDDING_TOWER_DEFAULTS = {
    "output_dim": 64,
    "dropout": 0.1,
    "normalize": True,
}

_EMBEDDING_REGISTRY: dict[str, tuple[type[torch.nn.Module], dict]] = {
    "tower": (EmbeddingTower, _EMBEDDING_TOWER_DEFAULTS),
}

_SCHED_REGISTRY: dict[str, tuple[type[LRSchedulerConfig], dict]] = {
    "plateau": (
        LRSchedulerConfig,
        {"factor": 0.1, "patience": 0, "min_lr": 1e-6, "threshold": 1e-6},
    ),
}


def _require_output_dim(module: torch.nn.Module, role: str) -> None:
    if not hasattr(module, "output_dim"):
        raise TypeError(f"{role} Module must expose an integer `output_dim` attribute")


def build_numeric_encoder(
    spec: str | torch.nn.Module,
    *,
    n_features: int,
    bins: list[torch.Tensor] | None = None,
) -> torch.nn.Module:
    """Build a numeric encoder from a spec string or pass through a Module.

    ``bins`` is required when ``spec`` resolves to the data-driven
    ``PiecewiseLinearEncoder`` (``num_encoder='ple'``); it is ignored
    otherwise. Use
    :meth:`scikit_rank.preprocessing.TabularPreprocessor.fit_ple_bins` to obtain
    bin edges from training data.
    """
    if isinstance(spec, torch.nn.Module):
        _require_output_dim(spec, "numeric encoder")
        return copy.deepcopy(spec)
    parsed = ModuleParserSpec(spec, allowed=_NUM_REGISTRY)
    name = parsed.module_name()
    cls, defaults = _NUM_REGISTRY[name]
    kwargs = {k: parsed.get(k, default=v) for k, v in defaults.items()}
    if name == "ple":
        if bins is None:
            raise ValueError(
                "num_encoder='ple' requires `bins`; pass them via"
                " build_numeric_encoder(..., bins=...) or rely on the"
                " sklearn estimator's ple_n_bins parameter.",
            )
        return cls(bins=bins, **kwargs)
    return cls(n_features=n_features, **kwargs)


def build_categorical_encoder(
    spec: str | torch.nn.Module,
    *,
    cardinalities: Sequence[int],
    embedding_dims: Sequence[int] | None = None,
) -> torch.nn.Module | None:
    """Build a categorical encoder; returns ``None`` if no categorical features."""
    if not cardinalities:
        return None
    if isinstance(spec, torch.nn.Module):
        _require_output_dim(spec, "categorical encoder")
        return copy.deepcopy(spec)
    parsed = ModuleParserSpec(spec, allowed=_CAT_REGISTRY)
    name = parsed.module_name()
    if name == "per_feature":
        if embedding_dims is None:
            raise ValueError("cat_encoder='per_feature' requires embedding_dims")
        return CategoricalEmbeddings(list(cardinalities), list(embedding_dims))
    # unified
    return UnifiedEmbeddings(
        list(cardinalities),
        **{k: parsed.get(k, default=v) for k, v in _UNIFIED_EMB_DEFAULTS.items()},
    )


def build_multihash_encoder(
    spec: str | torch.nn.Module,
    *,
    n_inputs: int,
) -> torch.nn.Module | None:
    """Build the shared-table multihash encoder, or ``None`` when there is none.

    ``cardinality`` and ``embedding_dim`` are read from the spec. ``n_hashes``
    may also live in the same spec for the estimator/preprocessor, while
    ``n_inputs`` stays data-derived (``n_features * n_hashes``) and is passed by
    the caller. Returns ``None`` when ``n_inputs == 0`` so a default spec with no
    multihash features wires nothing in.
    """
    if n_inputs <= 0:
        return None
    if isinstance(spec, torch.nn.Module):
        _require_output_dim(spec, "multihash encoder")
        return copy.deepcopy(spec)
    parsed = ModuleParserSpec(spec, allowed=_MULTIHASH_REGISTRY)
    cls, _defaults = _MULTIHASH_REGISTRY[parsed.module_name()]
    config = multihash_encoder_config(parsed)
    return cls(
        cardinality=config["cardinality"],
        n_inputs=n_inputs,
        embedding_dim=config["embedding_dim"],
    )


def multihash_encoder_config(spec: str | torch.nn.Module | ModuleParserSpec) -> dict[str, int]:
    """Resolve hashing/model config carried by ``multihash_encoder`` spec."""
    if isinstance(spec, torch.nn.Module):
        config = dict(_MULTIHASH_DEFAULTS)
    else:
        parsed = (
            spec
            if isinstance(spec, ModuleParserSpec)
            else ModuleParserSpec(spec, allowed=_MULTIHASH_REGISTRY)
        )
        _, defaults = _MULTIHASH_REGISTRY[parsed.module_name()]
        config = {k: parsed.get(k, default=v) for k, v in defaults.items()}
    out = {
        "cardinality": int(config["cardinality"]),
        "n_hashes": int(config["n_hashes"]),
        "embedding_dim": int(config["embedding_dim"]),
    }
    for key, value in out.items():
        if value <= 0:
            raise ValueError(f"multihash_encoder {key} must be positive, got {value}")
    return out


def build_embedding_encoder(
    spec: str | torch.nn.Module,
    *,
    input_dim: int,
) -> torch.nn.Module:
    """Build a dense external-embedding projection tower from a spec or Module.

    ``input_dim`` (the incoming vector width) is data-derived and passed as a
    scalar, mirroring ``n_features`` for the numeric encoder; the spec carries
    the projection ``output_dim`` / ``dropout`` / ``normalize`` tunables.
    """
    if isinstance(spec, torch.nn.Module):
        _require_output_dim(spec, "embedding encoder")
        return copy.deepcopy(spec)
    parsed = ModuleParserSpec(spec, allowed=_EMBEDDING_REGISTRY)
    cls, defaults = _EMBEDDING_REGISTRY[parsed.module_name()]
    kwargs = {k: parsed.get(k, default=v) for k, v in defaults.items()}
    return cls(input_dim=input_dim, **kwargs)


def _build_reference_layers(
    *,
    multihash_encoder: str | torch.nn.Module,
    multihash_n_inputs: int | None,
    embedding_encoders: dict[str, str | torch.nn.Module] | None,
    embedding_input_dims: dict[str, int] | None,
) -> dict[str, torch.nn.Module]:
    """Build the multihash + dense external-embedding ("reference") input streams.

    ``multihash_encoder`` carries cardinality/n_hashes/embedding_dim in one
    spec. The model only needs the parsed cardinality/embedding_dim plus the
    data-derived ``multihash_n_inputs`` supplied by the preprocessor.
    ``embedding_encoders`` maps one spec per dense stream, with its incoming
    vector width supplied in ``embedding_input_dims``. Returns the streams keyed
    by model input-layer name.
    """
    layers: dict[str, torch.nn.Module] = {}
    if multihash_n_inputs is not None:
        multihash = build_multihash_encoder(
            multihash_encoder,
            n_inputs=multihash_n_inputs,
        )
        if multihash is not None:
            layers["multihash"] = multihash
    if embedding_encoders:
        dims = embedding_input_dims or {}
        for name, embedding_spec in embedding_encoders.items():
            if name in layers:
                raise ValueError(f"embedding_encoders duplicate built-in stream: {name!r}")
            if name not in dims:
                raise ValueError(
                    f"embedding_input_dims is missing an input_dim for {name!r}",
                )
            layers[name] = build_embedding_encoder(embedding_spec, input_dim=dims[name])
    return layers


def build_dcnv2(
    *,
    n_num_features: int,
    cardinalities: Sequence[int],
    embedding_dims: Sequence[int] | None = None,
    hidden_units: Sequence[int] = (256, 128),
    cross_layers: int = 3,
    cross_rank: int | None = None,
    structure: Literal["stacked", "parallel"] = "stacked",
    num_encoder: str | torch.nn.Module = "identity",
    cat_encoder: str | torch.nn.Module = "per_feature",
    reducer: torch.nn.Module | None = None,
    n_outputs: int = 1,
    dropout: float = 0.0,
    activation: str = "relu",
    batch_norm: bool = False,
    gated_cross: bool = False,
    cross_type: str = "standard",
    mask_ratio: float = 0.5,
    use_moe: bool = False,
    num_experts: int = 4,
    moe_top_k: int = 2,
    use_inner_cross_layers: bool = False,
    use_pytorch_init: bool = False,
    use_coral_head: bool = False,
    num_encoder_bins: list[torch.Tensor] | None = None,
    multihash_encoder: str | torch.nn.Module = "multihash",
    multihash_n_inputs: int | None = None,
    embedding_encoders: dict[str, str | torch.nn.Module] | None = None,
    embedding_input_dims: dict[str, int] | None = None,
    extra_layers: dict[str, torch.nn.Module] | None = None,
) -> DCNv2:
    """Compose a :class:`DCNv2` from scalar / string hyperparams.

    The head is built internally against the computed post-cross/deep
    dimension: ``torch.nn.Linear(final_dim, n_outputs)`` by default, or
    :class:`~scikit_rank.model.CoralLayer(final_dim, n_outputs)` when
    ``use_coral_head=True`` (``n_outputs`` is interpreted as ``num_classes``
    in that case).
    """
    layers = {}
    if n_num_features > 0:
        layers["num"] = build_numeric_encoder(
            num_encoder,
            n_features=n_num_features,
            bins=num_encoder_bins,
        )
    if (
        cat := build_categorical_encoder(
            cat_encoder,
            cardinalities=cardinalities,
            embedding_dims=embedding_dims,
        )
    ) is not None:
        layers["cat"] = cat
    reference_layers = _build_reference_layers(
        multihash_encoder=multihash_encoder,
        multihash_n_inputs=multihash_n_inputs,
        embedding_encoders=embedding_encoders,
        embedding_input_dims=embedding_input_dims,
    )
    overlap = set(layers).intersection(reference_layers)
    if overlap:
        raise ValueError(f"reference streams duplicate built-in streams: {sorted(overlap)}")
    layers.update(reference_layers)
    if extra_layers:
        overlap = set(layers).intersection(extra_layers)
        if overlap:
            raise ValueError(f"extra_layers duplicate built-in streams: {sorted(overlap)}")
        for name, module in extra_layers.items():
            _require_output_dim(module, f"extra layer {name!r}")
            layers[name] = copy.deepcopy(module)
    if not layers:
        raise ValueError(
            "DCNv2 needs at least one input stream.",
        )

    reducer = reducer if reducer is not None else Concat(dim=-1)
    if not hasattr(reducer, "compute_output_dim"):
        raise TypeError("reducer must expose compute_output_dim(input_dims) -> int")

    input_dims = {name: m.output_dim() for name, m in layers.items()}
    rep_dim = reducer.compute_output_dim(input_dims)

    cross_network = CrossNetwork(
        rep_dim,
        cross_layers,
        rank=cross_rank,
        gated=gated_cross,
        cross_type=cross_type,
        mask_ratio=mask_ratio,
    )

    if structure not in ("stacked", "parallel"):
        raise ValueError(f"structure must be 'stacked' or 'parallel', got {structure!r}")

    if use_inner_cross_layers:
        if structure != "stacked":
            raise ValueError("use_inner_cross_layers requires structure='stacked'")
        if not hidden_units or len(hidden_units) < 2:
            raise ValueError(
                "use_inner_cross_layers requires hidden_units with len >= 2",
            )
        adapter_out_dim = hidden_units[0]
        inner_total = sum(cross_network.inner_dims())
        adapter = torch.nn.Linear(rep_dim, adapter_out_dim, bias=False)
        deep_network = DeepNetwork(
            adapter_out_dim + inner_total,
            list(hidden_units[1:]),
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            use_moe=use_moe,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
        )
        body = StackedCrossDeep(
            cross_network,
            deep_network,
            use_inner_cross_layers=True,
            adapter=adapter,
        )
        final_dim = deep_network.output_dim()
    else:
        deep_network = (
            DeepNetwork(
                rep_dim,
                list(hidden_units),
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm,
                use_moe=use_moe,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
            )
            if hidden_units
            else None
        )
        deep_out = deep_network.output_dim() if deep_network is not None else rep_dim
        if structure == "stacked":
            body = StackedCrossDeep(cross_network, deep_network)
            final_dim = deep_out
        else:
            body = ParallelCrossDeep(cross_network, deep_network)
            final_dim = rep_dim + deep_out

    head = (
        CoralLayer(in_features=final_dim, num_classes=n_outputs)
        if use_coral_head
        else torch.nn.Linear(final_dim, n_outputs)
    )

    model = DCNv2(layers=layers, reducer=reducer, body=body, head=head)
    if not use_pytorch_init:
        _init_weights(model)
    return model


def _finalnet_equal_output_chunks(
    encoder: torch.nn.Module,
    n_features: int,
) -> list[int]:
    """Split a built-in encoder's flat output into equal logical fields."""
    output_dim = encoder.output_dim()
    if output_dim % n_features:
        raise ValueError(
            "Built-in encoder output dimension is not divisible by its feature count: "
            f"{output_dim} % {n_features} != 0",
        )
    return [output_dim // n_features] * n_features


def _finalnet_numeric_output_chunks(
    spec: str | torch.nn.Module,
    encoder: torch.nn.Module,
    *,
    n_features: int,
    bins: list[torch.Tensor] | None,
) -> list[int]:
    """Return per-field widths for a FinalNet numeric encoder."""
    if isinstance(spec, torch.nn.Module):
        return [encoder.output_dim()]
    parsed = ModuleParserSpec(spec, allowed=_NUM_REGISTRY)
    if parsed.module_name() == "ple" and parsed.get("embedding_dim", default=None) is None:
        if bins is None:
            raise ValueError("PLE output chunks require fitted bins")
        return [len(edges) - 1 for edges in bins]
    return _finalnet_equal_output_chunks(encoder, n_features)


def _finalnet_categorical_output_chunks(
    spec: str | torch.nn.Module,
    encoder: torch.nn.Module,
    *,
    embedding_dims: Sequence[int],
    n_features: int,
) -> list[int]:
    """Return per-field widths for a FinalNet categorical encoder."""
    if isinstance(spec, torch.nn.Module):
        return [encoder.output_dim()]
    parsed = ModuleParserSpec(spec, allowed=_CAT_REGISTRY)
    if parsed.module_name() == "per_feature":
        return list(embedding_dims)
    return _finalnet_equal_output_chunks(encoder, n_features)


def _finalnet_reference_output_chunks(
    reference_layers: dict[str, torch.nn.Module],
    *,
    multihash_encoder: str | torch.nn.Module,
    multihash_n_inputs: int | None,
    embedding_encoders: dict[str, str | torch.nn.Module] | None,
) -> list[int]:
    """Return logical-field widths for FinalNet reference input streams."""
    output_chunks: list[int] = []
    if multihash := reference_layers.get("multihash"):
        if isinstance(multihash_encoder, torch.nn.Module):
            output_chunks.append(multihash.output_dim())
        else:
            multihash_config = multihash_encoder_config(multihash_encoder)
            n_inputs = int(multihash_n_inputs or 0)
            n_hashes = multihash_config["n_hashes"]
            if n_inputs % n_hashes:
                raise ValueError(
                    "multihash_n_inputs must be divisible by the configured n_hashes: "
                    f"{n_inputs} % {n_hashes} != 0",
                )
            output_chunks.extend(
                _finalnet_equal_output_chunks(multihash, n_inputs // n_hashes),
            )
    output_chunks.extend(reference_layers[name].output_dim() for name in embedding_encoders or {})
    return output_chunks


def _finalnet_output_chunks(
    layers: dict[str, torch.nn.Module],
    *,
    n_num_features: int,
    cardinalities: Sequence[int],
    embedding_dims: Sequence[int] | None,
    num_encoder: str | torch.nn.Module,
    cat_encoder: str | torch.nn.Module,
    num_encoder_bins: list[torch.Tensor] | None,
    multihash_encoder: str | torch.nn.Module,
    multihash_n_inputs: int | None,
    embedding_encoders: dict[str, str | torch.nn.Module] | None,
    extra_layers: dict[str, torch.nn.Module] | None,
) -> list[int]:
    """Resolve FinalNet logical-field widths only when field gating is enabled."""
    output_chunks: list[int] = []
    if n_num_features > 0:
        output_chunks.extend(
            _finalnet_numeric_output_chunks(
                num_encoder,
                layers["num"],
                n_features=n_num_features,
                bins=num_encoder_bins,
            ),
        )
    if cardinalities:
        output_chunks.extend(
            _finalnet_categorical_output_chunks(
                cat_encoder,
                layers["cat"],
                embedding_dims=list(embedding_dims or []),
                n_features=len(cardinalities),
            ),
        )
    output_chunks.extend(
        _finalnet_reference_output_chunks(
            layers,
            multihash_encoder=multihash_encoder,
            multihash_n_inputs=multihash_n_inputs,
            embedding_encoders=embedding_encoders,
        ),
    )
    if extra_layers:
        output_chunks.extend(layers[name].output_dim() for name in extra_layers)
    return output_chunks


def _build_finalnet_field_gate(
    *,
    enabled: bool,
    reducer: torch.nn.Module,
    representation_dim: int,
    output_chunks: Sequence[int],
) -> FinalNetFieldGate | None:
    if not enabled:
        return None
    if not isinstance(reducer, Concat):
        raise TypeError("FinalNet field gating requires the default Concat reducer")
    if len(set(output_chunks)) != 1:
        raise ValueError(
            "FinalNet field gating requires equal-width field representations, got "
            f"{list(output_chunks)}",
        )
    return FinalNetFieldGate(representation_dim, len(output_chunks))


def build_finalnet(
    *,
    n_num_features: int,
    cardinalities: Sequence[int],
    embedding_dims: Sequence[int] | None = None,
    block_type: str = "2B",
    block1_hidden_units: Sequence[int] = (400, 400),
    block2_hidden_units: Sequence[int] | None = None,
    block1_hidden_activations: str | Sequence[str | None] | None = None,
    block2_hidden_activations: str | Sequence[str | None] | None = None,
    block1_dropout: float | Sequence[float] = 0.0,
    block2_dropout: float | Sequence[float] | None = None,
    batch_norm: bool = True,
    residual_type: str = "concat",
    interaction_activation: str | None = "relu",
    use_field_gate: bool = False,
    use_pytorch_init: bool = False,
    num_encoder: str | torch.nn.Module = "identity",
    cat_encoder: str | torch.nn.Module = "per_feature",
    reducer: torch.nn.Module | None = None,
    n_outputs: int = 1,
    num_encoder_bins: list[torch.Tensor] | None = None,
    multihash_encoder: str | torch.nn.Module = "multihash",
    multihash_n_inputs: int | None = None,
    embedding_encoders: dict[str, str | torch.nn.Module] | None = None,
    embedding_input_dims: dict[str, int] | None = None,
    extra_layers: dict[str, torch.nn.Module] | None = None,
) -> FinalNet:
    """Compose a FINAL model from feature specs and block hyperparameters."""
    if block_type not in ("1B", "2B"):
        raise ValueError("block_type must be '1B' or '2B'")

    layers: dict[str, torch.nn.Module] = {}
    if n_num_features > 0:
        layers["num"] = build_numeric_encoder(
            num_encoder,
            n_features=n_num_features,
            bins=num_encoder_bins,
        )
    if (
        cat := build_categorical_encoder(
            cat_encoder,
            cardinalities=cardinalities,
            embedding_dims=embedding_dims,
        )
    ) is not None:
        layers["cat"] = cat
    reference_layers = _build_reference_layers(
        multihash_encoder=multihash_encoder,
        multihash_n_inputs=multihash_n_inputs,
        embedding_encoders=embedding_encoders,
        embedding_input_dims=embedding_input_dims,
    )
    overlap = set(layers).intersection(reference_layers)
    if overlap:
        raise ValueError(f"reference streams duplicate built-in streams: {sorted(overlap)}")
    layers.update(reference_layers)
    if extra_layers:
        overlap = set(layers).intersection(extra_layers)
        if overlap:
            raise ValueError(f"extra_layers duplicate built-in streams: {sorted(overlap)}")
        for name, module in extra_layers.items():
            _require_output_dim(module, f"extra layer {name!r}")
            layers[name] = copy.deepcopy(module)
    if not layers:
        raise ValueError("FinalNet needs at least one input stream.")

    reducer = reducer if reducer is not None else Concat(dim=-1)
    if not hasattr(reducer, "compute_output_dim"):
        raise TypeError("reducer must expose compute_output_dim(input_dims) -> int")
    representation_dim = reducer.compute_output_dim(
        {name: m.output_dim() for name, m in layers.items()},
    )
    output_chunks = (
        _finalnet_output_chunks(
            layers,
            n_num_features=n_num_features,
            cardinalities=cardinalities,
            embedding_dims=embedding_dims,
            num_encoder=num_encoder,
            cat_encoder=cat_encoder,
            num_encoder_bins=num_encoder_bins,
            multihash_encoder=multihash_encoder,
            multihash_n_inputs=multihash_n_inputs,
            embedding_encoders=embedding_encoders,
            extra_layers=extra_layers,
        )
        if use_field_gate
        else ()
    )
    field_gate = _build_finalnet_field_gate(
        enabled=use_field_gate,
        reducer=reducer,
        representation_dim=representation_dim,
        output_chunks=output_chunks,
    )

    first_units = list(block1_hidden_units)
    first_block = FinalBlock(
        representation_dim if field_gate is None else field_gate.output_dim(),
        first_units,
        hidden_activations=block1_hidden_activations,
        dropout=block1_dropout,
        batch_norm=batch_norm,
        residual_type=residual_type,
        interaction_activation=interaction_activation,
    )
    first_head = torch.nn.Linear(first_block.output_dim(), n_outputs)
    if block_type == "1B":
        model = FinalNet(
            layers=layers,
            reducer=reducer,
            block1=first_block,
            head1=first_head,
            field_gate=field_gate,
        )
    else:
        second_units = list(block2_hidden_units or block1_hidden_units)
        second_block = FinalBlock(
            representation_dim,
            second_units,
            hidden_activations=(
                block1_hidden_activations
                if block2_hidden_activations is None
                else block2_hidden_activations
            ),
            dropout=block1_dropout if block2_dropout is None else block2_dropout,
            batch_norm=batch_norm,
            residual_type=residual_type,
            interaction_activation=interaction_activation,
        )
        model = FinalNet(
            layers=layers,
            reducer=reducer,
            block1=first_block,
            head1=first_head,
            block2=second_block,
            head2=torch.nn.Linear(second_block.output_dim(), n_outputs),
            field_gate=field_gate,
        )
    if not use_pytorch_init:
        _init_weights(model)
    if field_gate is not None:
        # Field gating requires the reference zero-weight/unit-bias initialization.
        field_gate.reset_reference_parameters()
    return model


def _select_final_mlp_encoder_input(
    encoder: torch.nn.Module,
    indices: list[int],
    total: int,
) -> torch.nn.Module:
    return encoder if indices == list(range(total)) else InputSlice(encoder, indices)


def _build_final_mlp_layers(
    *,
    num_feature_names: Sequence[str],
    cat_feature_names: Sequence[str],
    cardinalities: Sequence[int],
    embedding_dims: Sequence[int],
    num_encoder: str | torch.nn.Module,
    cat_encoder: str | torch.nn.Module,
    num_encoder_bins: list[torch.Tensor] | None,
    multihash_feature_names: Sequence[str],
    multihash_encoder: str | torch.nn.Module,
    multihash_n_hashes: int,
    embedding_encoders: dict[str, str | torch.nn.Module] | None,
    embedding_input_dims: dict[str, int] | None,
    context: Sequence[str] | None,
) -> dict[str, torch.nn.Module]:
    """Build one FinalMLP encoder graph, optionally for a context subset."""
    selected = None if context is None else set(context)
    known = (
        set(num_feature_names)
        | set(cat_feature_names)
        | set(multihash_feature_names)
        | set(embedding_input_dims or {})
    )
    if selected is not None and (unknown := selected - known):
        raise ValueError(f"Unknown FinalMLP context features: {sorted(unknown)}")

    layers: dict[str, torch.nn.Module] = {}
    num_indices = [
        i for i, name in enumerate(num_feature_names) if selected is None or name in selected
    ]
    if num_indices:
        if isinstance(num_encoder, torch.nn.Module) and len(num_indices) != len(num_feature_names):
            raise ValueError(
                "A custom num_encoder cannot encode a subset of FinalMLP context features; "
                "select all numeric features or use a built-in encoder spec",
            )
        bins = [num_encoder_bins[i] for i in num_indices] if num_encoder_bins is not None else None
        encoder = build_numeric_encoder(num_encoder, n_features=len(num_indices), bins=bins)
        layers["num"] = _select_final_mlp_encoder_input(
            encoder,
            num_indices,
            len(num_feature_names),
        )

    cat_indices = [
        i for i, name in enumerate(cat_feature_names) if selected is None or name in selected
    ]
    if cat_indices:
        if isinstance(cat_encoder, torch.nn.Module) and len(cat_indices) != len(cat_feature_names):
            raise ValueError(
                "A custom cat_encoder cannot encode a subset of FinalMLP context features; "
                "select all categorical features or use a built-in encoder spec",
            )
        encoder = build_categorical_encoder(
            cat_encoder,
            cardinalities=[cardinalities[i] for i in cat_indices],
            embedding_dims=[embedding_dims[i] for i in cat_indices],
        )
        if encoder is not None:
            layers["cat"] = _select_final_mlp_encoder_input(
                encoder,
                cat_indices,
                len(cat_feature_names),
            )

    multihash_indices = [
        i for i, name in enumerate(multihash_feature_names) if selected is None or name in selected
    ]
    if multihash_indices:
        n_multihash_features = len(multihash_feature_names)
        if (
            isinstance(multihash_encoder, torch.nn.Module)
            and len(multihash_indices) != n_multihash_features
        ):
            raise ValueError(
                "A custom multihash_encoder cannot encode a subset of FinalMLP context "
                "features; select all multihash features or use a built-in encoder spec",
            )
        input_indices = [
            feature_idx * multihash_n_hashes + hash_idx
            for feature_idx in multihash_indices
            for hash_idx in range(multihash_n_hashes)
        ]
        encoder = build_multihash_encoder(multihash_encoder, n_inputs=len(input_indices))
        if encoder is not None:
            layers["multihash"] = _select_final_mlp_encoder_input(
                encoder,
                input_indices,
                len(multihash_feature_names) * multihash_n_hashes,
            )

    if embedding_input_dims:
        specs = embedding_encoders or {}
        for name, input_dim in embedding_input_dims.items():
            if selected is None or name in selected:
                spec = specs.get(name, "tower")
                layers[name] = build_embedding_encoder(spec, input_dim=input_dim)

    return layers


def build_final_mlp(
    *,
    num_feature_names: Sequence[str],
    cat_feature_names: Sequence[str],
    cardinalities: Sequence[int],
    embedding_dims: Sequence[int],
    mlp1_hidden_units: Sequence[int] = (64, 64, 64),
    mlp1_hidden_activations: str = "relu",
    mlp1_dropout: float = 0.0,
    mlp1_batch_norm: bool = False,
    mlp2_hidden_units: Sequence[int] = (64, 64, 64),
    mlp2_hidden_activations: str = "relu",
    mlp2_dropout: float = 0.0,
    mlp2_batch_norm: bool = False,
    use_fs: bool = True,
    fs_hidden_units: Sequence[int] = (64,),
    fs1_context: Sequence[str] = (),
    fs2_context: Sequence[str] = (),
    num_heads: int = 1,
    context_dim: int = 10,
    use_pytorch_init: bool = False,
    num_encoder: str | torch.nn.Module = "identity",
    cat_encoder: str | torch.nn.Module = "per_feature",
    num_encoder_bins: list[torch.Tensor] | None = None,
    multihash_feature_names: Sequence[str] = (),
    multihash_encoder: str | torch.nn.Module = "multihash",
    multihash_n_hashes: int = 2,
    embedding_encoders: dict[str, str | torch.nn.Module] | None = None,
    embedding_input_dims: dict[str, int] | None = None,
) -> FinalMLP:
    """Build FinalMLP on top of scikit-rank feature encoders."""
    if not mlp1_hidden_units or not mlp2_hidden_units:
        raise ValueError("FinalMLP requires non-empty mlp1_hidden_units and mlp2_hidden_units")
    if context_dim <= 0:
        raise ValueError(f"context_dim must be positive, got {context_dim}")

    encoder_kwargs = {
        "num_feature_names": num_feature_names,
        "cat_feature_names": cat_feature_names,
        "cardinalities": cardinalities,
        "embedding_dims": embedding_dims,
        "num_encoder": num_encoder,
        "cat_encoder": cat_encoder,
        "num_encoder_bins": num_encoder_bins,
        "multihash_feature_names": multihash_feature_names,
        "multihash_encoder": multihash_encoder,
        "multihash_n_hashes": multihash_n_hashes,
        "embedding_encoders": embedding_encoders,
        "embedding_input_dims": embedding_input_dims,
    }
    layers = _build_final_mlp_layers(**encoder_kwargs, context=None)
    if not layers:
        raise ValueError("FinalMLP needs at least one input stream.")
    reducer = Concat(dim=-1)
    feature_dim = reducer.compute_output_dim(
        {name: layer.output_dim() for name, layer in layers.items()},
    )
    feature_selection = None
    if use_fs:
        context1 = (
            _build_final_mlp_layers(**encoder_kwargs, context=fs1_context) if fs1_context else None
        )
        context2 = (
            _build_final_mlp_layers(**encoder_kwargs, context=fs2_context) if fs2_context else None
        )
        feature_selection = FeatureSelection(
            feature_dim,
            fs_hidden_units,
            context_dim,
            context_reducer=Concat(dim=-1),
            context1=context1,
            context2=context2,
        )

    mlp1 = DeepNetwork(
        feature_dim,
        list(mlp1_hidden_units),
        dropout=mlp1_dropout,
        activation=mlp1_hidden_activations,
        batch_norm=mlp1_batch_norm,
    )
    mlp2 = DeepNetwork(
        feature_dim,
        list(mlp2_hidden_units),
        dropout=mlp2_dropout,
        activation=mlp2_hidden_activations,
        batch_norm=mlp2_batch_norm,
    )
    model = FinalMLP(
        layers=layers,
        reducer=reducer,
        feature_selection=feature_selection,
        mlp1=mlp1,
        mlp2=mlp2,
        aggregation=InteractionAggregation(
            mlp1.output_dim(),
            mlp2.output_dim(),
            num_heads=num_heads,
            use_pytorch_init=use_pytorch_init,
        ),
    )
    if not use_pytorch_init:
        _init_weights(model)
    # w_xy is a raw parameter and retains its mode-specific constructor init.
    return model


def build_tabm(  # noqa: C901, PLR0912 - keep TabM-specific construction local
    *,
    num_feature_names: Sequence[str],
    cat_feature_names: Sequence[str],
    cardinalities: Sequence[int],
    embedding_dims: Sequence[int],
    n_outputs: int = 1,
    use_coral_head: bool = False,
    aggregation: EnsembleAggregation = "mean",
    n_blocks: int = 3,
    d_block: int = 512,
    dropout: float = 0.1,
    activation: str = "ReLU",
    k: int = 32,
    arch_type: Literal["tabm", "tabm-mini"] = "tabm",
    start_scaling_init: Literal["random-signs", "normal"] = "normal",
    use_pytorch_init: bool = False,
    num_encoder: str | torch.nn.Module = "identity",
    cat_encoder: str | torch.nn.Module = "per_feature",
    num_encoder_bins: list[torch.Tensor] | None = None,
    multihash_feature_names: Sequence[str] = (),
    multihash_encoder: str | torch.nn.Module = "multihash",
    multihash_n_hashes: int = 2,
    embedding_encoders: dict[str, str | torch.nn.Module] | None = None,
    embedding_input_dims: dict[str, int] | None = None,
) -> TabM:
    """Build TabM on top of scikit-rank feature encoders."""
    if arch_type not in ("tabm", "tabm-mini"):
        raise ValueError("arch_type must be 'tabm' or 'tabm-mini'")

    layers: dict[str, torch.nn.Module] = {}
    output_chunks: list[int] = []
    if num_feature_names:
        num = build_numeric_encoder(
            num_encoder,
            n_features=len(num_feature_names),
            bins=num_encoder_bins,
        )
        layers["num"] = num
        if isinstance(num_encoder, torch.nn.Module):
            output_chunks.append(num.output_dim())
        else:
            parsed_num_encoder = ModuleParserSpec(num_encoder, allowed=_NUM_REGISTRY)
            if (
                parsed_num_encoder.module_name() == "ple"
                and parsed_num_encoder.get("embedding_dim", default=None) is None
            ):
                if num_encoder_bins is None:
                    raise ValueError("PLE output chunks require fitted bins")
                output_chunks.extend(len(edges) - 1 for edges in num_encoder_bins)
            else:
                num_output_dim = num.output_dim()
                if num_output_dim % len(num_feature_names):
                    raise ValueError(
                        "Built-in numeric encoder output dimension is not divisible by "
                        f"its feature count: {num_output_dim} % {len(num_feature_names)} != 0",
                    )
                output_chunks.extend(
                    [num_output_dim // len(num_feature_names)] * len(num_feature_names),
                )
    if cat_feature_names:
        cat = build_categorical_encoder(
            cat_encoder,
            cardinalities=cardinalities,
            embedding_dims=embedding_dims,
        )
        if cat is not None:
            layers["cat"] = cat
            if isinstance(cat_encoder, torch.nn.Module):
                output_chunks.append(cat.output_dim())
            else:
                parsed_cat_encoder = ModuleParserSpec(cat_encoder, allowed=_CAT_REGISTRY)
                if parsed_cat_encoder.module_name() == "per_feature":
                    output_chunks.extend(embedding_dims)
                else:
                    cat_output_dim = cat.output_dim()
                    if cat_output_dim % len(cat_feature_names):
                        raise ValueError(
                            "Built-in categorical encoder output dimension is not divisible by "
                            f"its feature count: {cat_output_dim} % {len(cat_feature_names)} != 0",
                        )
                    output_chunks.extend(
                        [cat_output_dim // len(cat_feature_names)] * len(cat_feature_names),
                    )
    if multihash_feature_names:
        multihash = build_multihash_encoder(
            multihash_encoder,
            n_inputs=len(multihash_feature_names) * multihash_n_hashes,
        )
        if multihash is not None:
            layers["multihash"] = multihash
            if isinstance(multihash_encoder, torch.nn.Module):
                output_chunks.append(multihash.output_dim())
            else:
                multihash_output_dim = multihash.output_dim()
                if multihash_output_dim % len(multihash_feature_names):
                    raise ValueError(
                        "Built-in multihash encoder output dimension is not divisible by "
                        "its feature count: "
                        f"{multihash_output_dim} % {len(multihash_feature_names)} != 0",
                    )
                output_chunks.extend(
                    [multihash_output_dim // len(multihash_feature_names)]
                    * len(multihash_feature_names),
                )
    if embedding_input_dims:
        specs = embedding_encoders or {}
        for name, input_dim in embedding_input_dims.items():
            if name in layers:
                raise ValueError(f"embedding_encoders duplicate built-in stream: {name!r}")
            layer = build_embedding_encoder(specs.get(name, "tower"), input_dim=input_dim)
            layers[name] = layer
            output_chunks.append(layer.output_dim())
    if not layers:
        raise ValueError("TabM needs at least one input stream.")

    reducer = Concat(dim=-1)
    input_dim = reducer.compute_output_dim(
        {name: layer.output_dim() for name, layer in layers.items()},
    )
    registered_layers = torch.nn.ModuleDict(layers)
    # Preserve the package's specialized backbone/head initialization. Only the
    # feature encoders follow scikit-rank's existing initialization policy.
    if not use_pytorch_init:
        _init_weights(registered_layers)

    backbone = tabm_lib.make_tabm_backbone(
        d_in=input_dim,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        activation=activation,
        k=k,
        arch_type=arch_type,
        start_scaling_init=start_scaling_init,
        start_scaling_init_chunks=output_chunks,
    )
    output_dim = n_outputs - 1 if use_coral_head else n_outputs
    if output_dim <= 0:
        raise ValueError(f"TabM output dimension must be positive, got {output_dim}")
    head: torch.nn.Module = (
        CoralEnsemble(d_block, n_outputs, k=k)
        if use_coral_head
        else tabm_lib.LinearEnsemble(d_block, output_dim, k=k)
    )
    return TabM(
        layers=registered_layers,
        reducer=reducer,
        backbone=backbone,
        ensemble_view=tabm_lib.EnsembleView(k=k),
        head=head,
        aggregation=aggregation,
    )


def build_destine(
    *,
    n_num_features: int,
    cardinalities: Sequence[int],
    embedding_dim: int = 16,
    attention_dim: int = 16,
    num_heads: int = 2,
    attention_layers: int = 2,
    dnn_hidden_units: Sequence[int] = (),
    net_dropout: float = 0.0,
    attention_dropout: float = 0.0,
    activation: str = "relu",
    batch_norm: bool = False,
    relu_before_attention: bool = False,
    scale_attention: bool = True,
    unary_mode: Literal["paper", "static"] = "paper",
    residual_mode: str | None = "each_layer",
    attention_activation: bool = True,
    use_wide: bool = False,
    cat_encoder: str | torch.nn.Module = "per_feature",
    multihash_encoder: str | torch.nn.Module = "multihash",
    multihash_n_inputs: int | None = None,
    embedding_encoders: dict[str, str | torch.nn.Module] | None = None,
    embedding_input_dims: dict[str, int] | None = None,
    n_outputs: int = 1,
    use_coral_head: bool = False,
) -> DESTINE:
    """Build a DESTINE model against the same preprocessed streams as DCNv2.

    Every original numeric or categorical column becomes one equally-sized
    feature field. Multiple hashes of one source column are averaged back into
    one field, while each named dense embedding stream is represented as one
    additional field.
    """
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
    layers: dict[str, torch.nn.Module] = {}
    if n_num_features:
        layers["num"] = NumericFieldEmbeddings(n_num_features, embedding_dim)

    if cardinalities:
        if isinstance(cat_encoder, torch.nn.Module):
            categorical = copy.deepcopy(cat_encoder)
        else:
            parsed = ModuleParserSpec(cat_encoder, allowed=_CAT_REGISTRY)
            if parsed.module_name() == "per_feature":
                categorical = CategoricalEmbeddings(
                    list(cardinalities),
                    [embedding_dim] * len(cardinalities),
                )
            else:
                requested_dim = parsed.get("embedding_dim", default=embedding_dim)
                if requested_dim != embedding_dim:
                    raise ValueError(
                        "DESTINE requires one common field width: cat_encoder "
                        f"embedding_dim={requested_dim} differs from embedding_dim={embedding_dim}",
                    )
                categorical = UnifiedEmbeddings(list(cardinalities), embedding_dim)
        layers["cat"] = ReshapedFieldEncoder(
            categorical,
            n_fields=len(cardinalities),
            embedding_dim=embedding_dim,
        )

    multihash_config = multihash_encoder_config(multihash_encoder)
    if multihash_n_inputs:
        multihash = build_multihash_encoder(
            multihash_encoder,
            n_inputs=multihash_n_inputs,
        )
        assert multihash is not None
        layers["multihash"] = MultiHashFieldEncoder(
            multihash,
            n_inputs=multihash_n_inputs,
            n_hashes=multihash_config["n_hashes"],
            input_embedding_dim=multihash_config["embedding_dim"],
            embedding_dim=embedding_dim,
        )

    if embedding_encoders:
        dims = embedding_input_dims or {}
        for name, spec in embedding_encoders.items():
            if name not in dims:
                raise ValueError(f"embedding_input_dims is missing input_dim for {name!r}")
            layers[name] = DenseFieldEncoder(
                build_embedding_encoder(spec, input_dim=dims[name]),
                embedding_dim=embedding_dim,
            )

    if not layers:
        raise ValueError("DESTINE needs at least one input stream")
    n_fields = sum(int(layer.n_fields()) for layer in layers.values())
    attention_output_dim = n_fields * attention_dim
    head = (
        CoralLayer(in_features=attention_output_dim, num_classes=n_outputs)
        if use_coral_head
        else torch.nn.Linear(attention_output_dim, n_outputs)
    )
    output_width = int(head.output_dim()) if hasattr(head, "output_dim") else n_outputs

    dnn = None
    dnn_head = None
    if dnn_hidden_units:
        dnn = DeepNetwork(
            n_fields * embedding_dim,
            list(dnn_hidden_units),
            dropout=net_dropout,
            activation=activation,
            batch_norm=batch_norm,
        )
        dnn_head = torch.nn.Linear(dnn.output_dim(), output_width)

    wide = (
        DESTINEWide(
            n_num_features=n_num_features,
            cardinalities=cardinalities,
            n_outputs=output_width,
            multihash_cardinality=(multihash_config["cardinality"] if multihash_n_inputs else None),
            multihash_n_inputs=multihash_n_inputs or 0,
            multihash_n_hashes=multihash_config["n_hashes"],
            embedding_input_dims=embedding_input_dims,
        )
        if use_wide
        else None
    )
    if attention_layers <= 0:
        raise ValueError("attention_layers must be positive")
    if residual_mode not in ("each_layer", "last_layer", "none", None):
        raise ValueError("residual_mode must be 'each_layer', 'last_layer', 'none', or None")
    attention = [
        DisentangledSelfAttention(
            embedding_dim if index == 0 else attention_dim,
            attention_dim,
            num_heads,
            attention_dropout,
            use_residual=residual_mode == "each_layer",
            use_scale=scale_attention,
            relu_before_attention=relu_before_attention,
            unary_mode=unary_mode,
        )
        for index in range(attention_layers)
    ]
    last_residual = (
        torch.nn.Linear(embedding_dim, attention_dim) if residual_mode == "last_layer" else None
    )
    model = DESTINE(
        layers=layers,
        attention=attention,
        last_residual=last_residual,
        attention_activation=attention_activation,
        head=head,
        dnn=dnn,
        dnn_head=dnn_head,
        wide=wide,
    )
    _init_weights(model)
    return model


def _xavier_normal_per_feature_(weight: torch.Tensor, gain: float = 1.0) -> None:
    """Xavier-normal init for a packed per-feature ``EinMix`` weight.

    The weight of an ``EinMix("b n i -> b n o", weight_shape="n i o")`` is a 3-D
    tensor ``(n_features, i, o)`` holding one independent ``i -> o`` linear map per
    feature. ``torch.nn.init.xavier_normal_`` would compute fan over the *packed*
    shape (folding ``n_features`` into ``fan_out``), which under-scales the init and
    makes it depend on ``n_features``. Instead, init each slice like an independent
    ``nn.Linear(i, o)`` -- ``std = gain * sqrt(2 / (i + o))`` -- so the scale is
    ``n_features``-independent. The last two dims are the linear map ``(i, o)``;
    ``sqrt(2 / (fan_in + fan_out))`` is symmetric in the two, so the ordering does
    not matter (also correct for a 2-D weight).
    """
    fan_in, fan_out = weight.shape[-2], weight.shape[-1]
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight.normal_(0.0, std)


def _init_weights(model: torch.nn.Module) -> None:
    """Re-initialize model weights in place (post-composition).

    Embeddings ~ ``N(0, 1e-4)``; ``Linear`` weights via Xavier-normal with zero
    bias; per-feature ``EinMix`` numeric encoders (``LinearNumericEncoder``,
    ``ple``, ``plr``) via :func:`_xavier_normal_per_feature_` -- each ``(i -> o)``
    slice is initialized like an independent ``nn.Linear(i, o)`` so its std is
    ``n_features``-independent -- with zero bias; the cross-layer additive bias is
    zeroed (overriding :class:`CrossLayer`'s uniform default).
    """
    for mod in model.modules():
        if isinstance(mod, NumericFieldEmbeddings):
            mod.reset_parameters()
        elif isinstance(mod, torch.nn.Embedding):
            torch.nn.init.normal_(mod.weight, mean=0.0, std=1e-4)
        elif isinstance(mod, EinMix):
            if getattr(mod, "weight", None) is not None:
                _xavier_normal_per_feature_(mod.weight)
            if getattr(mod, "bias", None) is not None:
                torch.nn.init.zeros_(mod.bias)
        elif isinstance(mod, torch.nn.Linear):
            if getattr(mod, "weight", None) is not None:
                torch.nn.init.xavier_normal_(mod.weight)
            if getattr(mod, "bias", None) is not None:
                torch.nn.init.zeros_(mod.bias)
        elif isinstance(mod, CrossLayer):
            torch.nn.init.zeros_(mod._bias)  # noqa: SLF001


def build_lr_scheduler_config(spec: str | None) -> LRSchedulerConfig | None:
    """Build an :class:`~scikit_rank.train.optimizers.LRSchedulerConfig` from a spec string.

    Mirrors the encoder builders (parse ``name[:k=v;...]`` against a registry), but
    returns a *config* rather than a Module: the scheduler needs the optimizer,
    which ``TrainingRun`` owns, so it is constructed there via
    :func:`~scikit_rank.train.optimizers.build_lr_scheduler`. ``None`` -> no scheduler.
    """
    if spec is None:
        return None
    parsed = ModuleParserSpec(spec, allowed=_SCHED_REGISTRY)
    name = parsed.module_name()
    cls, defaults = _SCHED_REGISTRY[name]
    kwargs = {k: parsed.get(k, default=v) for k, v in defaults.items()}
    return cls(scheduler_type=name, **kwargs)

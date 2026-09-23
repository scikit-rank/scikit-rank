"""FinalMLP backbone with feature selection and interaction aggregation."""

from collections.abc import Iterator, Sequence

import torch

from scikit_rank.modules.dcn import DeepNetwork


class InputSlice(torch.nn.Module):
    """Select input columns before applying an encoder."""

    def __init__(self, encoder: torch.nn.Module, indices: Sequence[int]) -> None:
        super().__init__()
        self._encoder = encoder
        self.register_buffer("_indices", torch.tensor(indices, dtype=torch.long))

    def output_dim(self) -> int:
        return self._encoder.output_dim()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x.index_select(1, self._indices))


def _encode_streams(
    layers: torch.nn.ModuleDict,
    reducer: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    encoded = {name: layer(inputs[name]) for name, layer in layers.items()}
    return reducer(encoded)


class _Gate(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_units: Sequence[int]) -> None:
        super().__init__()
        self._hidden = DeepNetwork(input_dim, list(hidden_units), activation="relu")
        self._output = torch.nn.Linear(self._hidden.output_dim(), output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self._output(self._hidden(x)))


class FeatureSelection(torch.nn.Module):
    """Produce independently gated inputs for the two FinalMLP streams."""

    def __init__(
        self,
        feature_dim: int,
        hidden_units: Sequence[int],
        context_dim: int,
        context_reducer: torch.nn.Module,
        context1: dict[str, torch.nn.Module] | None = None,
        context2: dict[str, torch.nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self._context_reducer = context_reducer
        self._context1 = torch.nn.ModuleDict(context1) if context1 is not None else None
        self._context2 = torch.nn.ModuleDict(context2) if context2 is not None else None
        self._context1_bias = (
            torch.nn.Parameter(torch.zeros(1, context_dim)) if context1 is None else None
        )
        self._context2_bias = (
            torch.nn.Parameter(torch.zeros(1, context_dim)) if context2 is None else None
        )
        input1_dim = (
            context_dim
            if context1 is None
            else context_reducer.compute_output_dim(
                {name: layer.output_dim() for name, layer in context1.items()},
            )
        )
        input2_dim = (
            context_dim
            if context2 is None
            else context_reducer.compute_output_dim(
                {name: layer.output_dim() for name, layer in context2.items()},
            )
        )
        self._gate1 = _Gate(input1_dim, feature_dim, hidden_units)
        self._gate2 = _Gate(input2_dim, feature_dim, hidden_units)

    def context_layers(
        self,
    ) -> tuple[torch.nn.ModuleDict | None, torch.nn.ModuleDict | None]:
        return self._context1, self._context2

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        flat_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = flat_emb.size(0)
        context1 = (
            self._context1_bias.expand(batch_size, -1)
            if self._context1 is None
            else _encode_streams(self._context1, self._context_reducer, inputs)
        )
        context2 = (
            self._context2_bias.expand(batch_size, -1)
            if self._context2 is None
            else _encode_streams(self._context2, self._context_reducer, inputs)
        )
        return flat_emb * self._gate1(context1), flat_emb * self._gate2(context2)


class InteractionAggregation(torch.nn.Module):
    """Multi-head bilinear fusion from the FinalMLP paper."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        num_heads: int = 1,
        *,
        use_pytorch_init: bool = False,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if x_dim % num_heads != 0 or y_dim % num_heads != 0:
            raise ValueError(
                "Final MLP tower dimensions must be divisible by num_heads: "
                f"x_dim={x_dim}, y_dim={y_dim}, num_heads={num_heads}",
            )
        self._num_heads = num_heads
        self._head_x_dim = x_dim // num_heads
        self._head_y_dim = y_dim // num_heads
        self._w_x = torch.nn.Linear(x_dim, 1)
        self._w_y = torch.nn.Linear(y_dim, 1)
        self._w_xy = torch.nn.Parameter(
            torch.empty(num_heads, self._head_x_dim, self._head_y_dim),
        )
        if use_pytorch_init:
            bound = self._w_xy.numel() ** -0.5
            torch.nn.init.uniform_(self._w_xy, -bound, bound)
        else:
            torch.nn.init.xavier_normal_(self._w_xy)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        head_x = x.reshape(-1, self._num_heads, self._head_x_dim)
        head_y = y.reshape(-1, self._num_heads, self._head_y_dim)
        bilinear = torch.einsum("bhi,hij,bhj->bh", head_x, self._w_xy, head_y).sum(dim=1)
        return self._w_x(x).squeeze(-1) + self._w_y(y).squeeze(-1) + bilinear


class FinalMLP(torch.nn.Module):
    """Two-stream MLP with feature selection and bilinear aggregation."""

    def __init__(
        self,
        *,
        layers: dict[str, torch.nn.Module],
        reducer: torch.nn.Module,
        mlp1: DeepNetwork,
        mlp2: DeepNetwork,
        aggregation: InteractionAggregation,
        feature_selection: FeatureSelection | None = None,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must contain at least one entry")
        self._layers = torch.nn.ModuleDict(layers)
        self._reducer = reducer
        self._feature_selection = feature_selection
        self._mlp1 = mlp1
        self._mlp2 = mlp2
        self._aggregation = aggregation

    def layers(self) -> torch.nn.ModuleDict:
        return self._layers

    def reducer(self) -> torch.nn.Module:
        return self._reducer

    def head(self) -> InteractionAggregation:
        return self._aggregation

    def embedding_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self._layers.parameters()
        if self._feature_selection is not None:
            for context in self._feature_selection.context_layers():
                if context is not None:
                    yield from context.parameters()

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        flat_emb = _encode_streams(self._layers, self._reducer, inputs)
        if self._feature_selection is None:
            feature1 = feature2 = flat_emb
        else:
            feature1, feature2 = self._feature_selection(inputs, flat_emb)
        return self._aggregation(self._mlp1(feature1), self._mlp2(feature2))

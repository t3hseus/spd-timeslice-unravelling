import torch
import gin
from torch import nn


class FFN(nn.Module):
    """FFN layer following Transformers and MLP mixer architectures"""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(model_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, model_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, model_dim: int, sublayer: nn.Module):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor):
        "Apply residual connection to any sublayer with the same size."
        return x + self.sublayer(self.norm(x))


@gin.configurable
class TrackEmbedder(nn.Module):
    def __init__(
        self,
        input_dim: int = 105,  # n_stations * 3 coords
        n_blocks: int = 2,
        model_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 32,
        dropout: float = 0.1
    ):
        super(TrackEmbedder, self).__init__()
        self.projection_layer = nn.Linear(input_dim, model_dim)
        self.blocks = nn.Sequential(
            *[SublayerConnection(
                model_dim, FFN(model_dim, hidden_dim, dropout))
              for _ in range(n_blocks)]
        )
        self.layer_norm = nn.LayerNorm(model_dim)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.projection_layer(x)
        x = self.blocks(x)
        x = self.layer_norm(x)
        x = self.output_layer(x)
        return x

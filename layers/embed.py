import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Project raw telemetry features into the model embedding dimension."""

    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.token_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=embed_dim,
            kernel_size=3,
            padding=padding,
            padding_mode="zeros",
        )

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x.reshape(B, T, N, -1)


class PositionalEmbedding(nn.Module):
    """Sinusoidal temporal position encoding for `(B, T, N, C)` inputs."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        self.embed_dim = embed_dim

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        ).exp()

        pe = torch.zeros(max_len, embed_dim).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = x.shape
        out = self.pe[:, :T].repeat(B * N, 1, 1)
        return out.reshape(B, N, T, self.embed_dim).permute(0, 2, 1, 3)


class FixedEmbedding(nn.Module):
    """Fixed sinusoidal embedding for calendar time fields."""

    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        position = torch.arange(0, num_features).float().unsqueeze(1)
        div_term = (
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        ).exp()

        weights = torch.zeros(num_features, embed_dim).float()
        weights[:, 0::2] = torch.sin(position * div_term)
        weights[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(num_features, embed_dim)
        self.emb.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x).detach()


class TimeFeatureEmbedding(nn.Module):
    """Linear embedding for continuous time features."""

    def __init__(self, embed_dim: int, embed_type: str = "timeF", freq: str = "h"):
        super().__init__()
        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        self.embed = nn.Linear(freq_map[freq], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class TemporalEmbedding(nn.Module):
    """Calendar embedding used when explicit time marks are provided."""

    def __init__(self, embed_dim: int, embed_type: str = "fixed", freq: str = "h"):
        super().__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        embed_class = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = embed_class(minute_size, embed_dim)
        self.hour_embed = embed_class(hour_size, embed_dim)
        self.weekday_embed = embed_class(weekday_size, embed_dim)
        self.day_embed = embed_class(day_size, embed_dim)
        self.month_embed = embed_class(month_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding(nn.Module):
    """
    Embed a telemetry tensor for M3RL.

    M3RL passes metrics, logs, and traces as `(B, T, N, C)` tensors. This module
    combines value projection with temporal position encoding and optional time
    marks, then returns `(B, T, N, D)`.
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.value_embedding = TokenEmbedding(
            num_features=num_features, embed_dim=embed_dim
        )
        self.position_embedding = PositionalEmbedding(embed_dim=embed_dim)
        self.temporal_embedding = (
            TemporalEmbedding(embed_dim=embed_dim, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(
                embed_dim=embed_dim, embed_type=embed_type, freq=freq
            )
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None):
        if x_mark is not None:
            x = (
                self.value_embedding(x)
                + self.position_embedding(x)
                + self.temporal_embedding(x_mark)
            )
        else:
            x = self.value_embedding(x) + self.position_embedding(x).to(x.device)

        return self.dropout(x)

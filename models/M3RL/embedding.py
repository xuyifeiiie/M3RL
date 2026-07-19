import torch
import torch.nn as nn
from typing import Tuple
from layers.embed import DataEmbedding
import math


class PositionalEmbedding1D(nn.Module):
    """Sinusoidal time encoding shared by all telemetry modalities."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :].unsqueeze(2)


class SpatioTemporalEmbedding(nn.Module):
    """
    Projects one modality into the shared node-time representation space.

    Each telemetry value is enriched with a time position and a learnable node
    identity, matching the paper's need to preserve both temporal order and
    microservice identity before fusion.
    """

    def __init__(self, num_features, num_nodes, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.value_projection = nn.Linear(num_features, embed_dim)

        self.time_embedding = PositionalEmbedding1D(
            embed_dim=embed_dim, max_len=max_len
        )

        self.node_embedding = nn.Embedding(
            num_embeddings=num_nodes, embedding_dim=embed_dim
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, N, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, N, embed_dim).
        """
        B, T, N, C = x.shape
        device = x.device

        val_emb = self.value_projection(x)
        time_emb = self.time_embedding(x)
        node_ids = torch.arange(N, device=device)
        node_emb = self.node_embedding(node_ids)
        node_emb = node_emb.unsqueeze(0).unsqueeze(0)

        final_emb = val_emb + time_emb + node_emb
        return self.dropout(self.norm(final_emb))


class EmbeddingLayer(nn.Module):
    """
    Encodes metrics, logs, and traces into a unified embedding dimension.

    Input tensors follow `(B, T, N, C_modality)` and outputs follow
    `(B, T, N, D)`, so later fusion and STNet blocks can operate on a common
    representation across modalities.
    """

    def __init__(self, settings, device):
        super(EmbeddingLayer, self).__init__()
        self.settings = settings
        self.device = device
        self.emb_dim = settings.emb_dim
        self.hidden_dim = settings.hidden_dim
        self.num_features = settings.num_features
        self.num_nodes = settings.num_nodes
        self.trace_dim = settings.trace_dim
        self.embed = self.emb_dim
        self.freq = settings.freq
        self.dropout = settings.dropout
        self.metric_embed = SpatioTemporalEmbedding(
            self.num_features, self.num_nodes, self.emb_dim
        )
        self.trace_embed = SpatioTemporalEmbedding(
            self.trace_dim, self.num_nodes, self.emb_dim
        )

        if settings.if_semantics and settings.if_log_feature:
            self.log_dim = settings.log_dim + settings.num_events
        elif settings.if_semantics:
            self.log_dim = settings.log_dim
        elif settings.if_log_feature:
            self.log_dim = settings.num_events
        self.log_embed = SpatioTemporalEmbedding(
            self.log_dim, self.num_nodes, self.emb_dim
        )

        self.dropout_layer = nn.Dropout(settings.dropout)

    def forward(
        self, metrics: torch.Tensor, logs: torch.Tensor, traces: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the embedding layer.

        Args:
            metrics (torch.Tensor): Input metrics tensor (B, T, N, Cm).
            logs (torch.Tensor): Input logs tensor (B, T, N, Cl).
            traces (torch.Tensor): Input traces tensor (B, T, N, Ct).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Embedded tensors
                - metric_embedded (B, T, N, D').
                - log_embedded (B, T, N, D').
                - trace_embedded (B, T, N, D').
        """
        metric_embedded = self.metric_embed(metrics)
        log_embedded = self.log_embed(logs)
        trace_embedded = self.trace_embed(traces)

        return metric_embedded, log_embedded, trace_embedded

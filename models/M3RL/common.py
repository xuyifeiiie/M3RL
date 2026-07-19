import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from wandb.sdk.internal.system.assets.aggregators import aggregate_last


class TemporalAttentionPooling(nn.Module):
    """
    Aggregates the time dimension (T) of a [B, T, N, D] tensor to produce a [B, N, D] tensor.
    It learns a set of attention weights for each node over the time steps.
    """

    def __init__(self, input_dim: int, hidden_dim_multiplier: int = 2):
        """
        Args:
            input_dim (int): The feature dimension D of the input tensor.
            hidden_dim_multiplier (int, optional): Multiplier for the hidden layer size. Defaults to 2.
        """
        super().__init__()
        self.input_dim = input_dim

        # This MLP computes the attention scores (unnormalized)
        # Input: [B*N, T, D], Output: [B*N, T, 1]
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim * hidden_dim_multiplier),
            nn.GELU(),
            nn.Linear(input_dim * hidden_dim_multiplier, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal attention pooling.
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, N, D].
        Returns:
            torch.Tensor: Output tensor of shape [B, N, D].
        """
        B, T, N, D = x.shape

        # Reshape for efficient batch processing: [B, T, N, D] -> [B*N, T, D]
        # We want to compute attention scores over T for each node independently
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # 1. Compute attention scores
        # attn_weights shape: [B*N, T, 1]
        attn_weights = self.attention_net(x_reshaped)

        # 2. Normalize scores using softmax
        # attn_weights_softmax shape: [B*N, T, 1]
        attn_weights_softmax = torch.softmax(attn_weights, dim=1)

        # 3. Apply attention weights (weighted sum)
        # We need to compute Σ (alpha_t * value_t)
        # (weights.transpose * values) is an efficient way to do this with bmm
        # attn_weights_softmax.transpose(1, 2) shape: [B*N, 1, T]
        # x_reshaped shape: [B*N, T, D]
        # weighted_repr shape: [B*N, 1, D]
        weighted_repr = torch.bmm(attn_weights_softmax.transpose(1, 2), x_reshaped)

        # Squeeze the middle dimension: [B*N, 1, D] -> [B*N, D]
        weighted_repr = weighted_repr.squeeze(1)

        # 4. Reshape back to the desired output format: [B*N, D] -> [B, N, D]
        node_representations = weighted_repr.view(B, N, D)

        return node_representations


class SpatialAttentionPooling(nn.Module):
    """Aggregates the node dimension (N) of a [B, N, D] tensor to produce a [B, D] tensor."""

    def __init__(self, input_dim: int, hidden_dim_multiplier: int = 2):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim * hidden_dim_multiplier),
            nn.Tanh(),
            nn.Linear(input_dim * hidden_dim_multiplier, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, N, D]
        Output: [B, D]
        """
        # attn_weights: [B, N, 1]
        attn_weights = self.attention_net(x)
        attn_weights_softmax = torch.softmax(attn_weights, dim=1)

        # weighted_repr: [B, 1, D]
        weighted_repr = torch.bmm(attn_weights_softmax.transpose(1, 2), x)

        return weighted_repr.squeeze(1)  # [B, D]


class AggregationModule(nn.Module):
    """Aggregates BTND representation to BD representation"""

    def __init__(self, input_dim, output_dim, timesteps=0, num_nodes=0, method="mean"):
        super(AggregationModule, self).__init__()
        self.method = method

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, output_dim),
        )

        if self.method == "flatten_mlp":
            self.flatten_mlp = nn.Sequential(
                nn.Linear(input_dim * timesteps * num_nodes, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )

    def forward(self, x_btnd):
        # x_bntd: (B, T, N, D_in)
        if self.method == "mean":
            # Mean pool over T and N dimensions
            aggregated = x_btnd.mean(dim=(1, 2))  # (B, D_in)
        elif self.method == "last_time_mean_node":
            # Take last time step, then mean pool over nodes
            aggregated = x_btnd[:, -1, :, :].mean(dim=1)  # (B, D_in)
        elif self.method == "flatten_mlp":
            B, T, N, D_in = x_btnd.shape
            flattened = x_btnd.reshape(B, -1)  # (B, T*N*D_in)
            aggregated = self.flatten_mlp(flattened)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

        output = self.mlp(aggregated)  # (B, D_out) = (B, D_sys)
        return output


class DeaggregationModule(nn.Module):
    """Expands BD representation back to BTND representation"""

    def __init__(self, input_dim, output_dim, timesteps=0, num_nodes=0):
        super(DeaggregationModule, self).__init__()
        self.T = timesteps
        self.N = num_nodes
        self.output_dim = output_dim  # Should match decoder STNet input dim (D_fuse)

        # MLP to expand features to the target flattened size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, self.T * self.N * output_dim),
        )

    def forward(self, x_bd):
        # x_bd: (B, D_sys)
        B, _ = x_bd.shape
        expanded_flat = self.mlp(x_bd)  # (B, T*N*D_out)
        # Reshape to (B, T, N, D_out)
        output_btnd = expanded_flat.view(B, self.T, self.N, self.output_dim)
        return output_btnd


class DeaggregationModuleStructured(nn.Module):
    """Expands BND representation to BTND representation using learnable temporal queries."""

    def __init__(self, input_dim: int, output_dim: int, timesteps: int):
        super().__init__()
        self.T = timesteps
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Learnable temporal queries, one query per reconstructed time step.
        self.temporal_queries = nn.Parameter(torch.randn(1, timesteps, input_dim))

        # Decode the concatenated node representation and temporal query.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, output_dim),
        )

        # 3. LayerNorm for stability
        self.norm_node = nn.LayerNorm(input_dim)
        self.norm_query = nn.LayerNorm(input_dim)

    def forward(self, x_bnd: torch.Tensor) -> torch.Tensor:
        # x_bnd: [B, N, D]
        B, N, D = x_bnd.shape

        # --- Prepare inputs ---
        # Repeat each node representation across all reconstructed time steps.
        # [B, N, D] -> [B, N, 1, D] -> [B, N, T, D]
        node_repr_expanded = (
            self.norm_node(x_bnd).unsqueeze(2).expand(-1, -1, self.T, -1)
        )

        # Repeat temporal queries for each batch item and node:
        # [1, T, D] -> [B, N, T, D]
        temp_queries_expanded = self.norm_query(self.temporal_queries).expand(
            B, N, -1, -1
        )

        # --- Combine and decode ---
        # Concatenate node representations with temporal queries.
        # [B, N, T, 2*D]
        combined_input = torch.cat([node_repr_expanded, temp_queries_expanded], dim=-1)

        # Decode with the MLP.
        # [B, N, T, D_out]
        output_bntd = self.mlp(combined_input)

        # Restore [B, T, N, D_out].
        output_btnd = output_bntd.permute(0, 2, 1, 3)

        return output_btnd


class DeaggregationModuleTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, timesteps: int):
        super().__init__()
        self.T = timesteps
        self.temporal_queries = nn.Parameter(torch.randn(timesteps, d_model))

        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, x_bnd: torch.Tensor) -> torch.Tensor:
        # x_bnd: [B, N, D]
        B, N, D = x_bnd.shape

        # memory: [B*N, 1, D]
        memory = x_bnd.reshape(B * N, 1, D)

        # tgt: [B*N, T, D]
        tgt = self.temporal_queries.unsqueeze(0).expand(B * N, -1, -1)

        output_bntd = self.transformer_decoder(tgt, memory)

        # [B*N, T, D] -> [B, N, T, D] -> [B, T, N, D]
        return output_bntd.view(B, N, self.T, D).permute(0, 2, 1, 3)

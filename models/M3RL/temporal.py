import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class SelfAttentionBlock(nn.Module):
    """Transformer encoder block used for temporal service representations."""

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff=None,
        dropout=0.1,
        activation="relu",
        batch_first=True,
    ):
        super(SelfAttentionBlock, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = src + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x


class TemporalSelfAttentionLayer(nn.Module):
    """
    Applies self-attention across the time dimension T.
    Input: (B, T, N, D')
    Output: (B, T, N, D')
    """

    def __init__(self, settings, device):
        super(TemporalSelfAttentionLayer, self).__init__()
        self.settings = settings
        self.device = device
        self.d_model = settings.d_model
        self.num_heads = settings.num_heads
        self.dropout = settings.dropout
        self.d_ff = self.d_model * 4

        self.self_attention_block = SelfAttentionBlock(
            self.d_model, self.num_heads, self.d_ff, self.dropout, batch_first=True
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor from previous layer (B, T, N, D').

        Returns:
            torch.Tensor: Output tensor after temporal self-attention (B, T, N, D').
        """
        B, T, N, D_prime = x.shape
        BN = B * N

        input_reshaped = x.permute(0, 2, 1, 3).reshape(BN, T, D_prime)

        output_reshaped = self.self_attention_block(input_reshaped)

        output_btn = output_reshaped.reshape(B, N, T, D_prime).permute(0, 2, 1, 3)

        return output_btn


class ShiftWindowAttention(nn.Module):
    """
    Shift-window self-attention along the temporal dimension.

    Input and output shape: (B, T, N, D).
    """

    def __init__(self, dim, num_heads=4, window_size=8, shift_size=4):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)

        pad_len = (self.window_size - T % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        _, L, _ = x.shape
        windows = L // self.window_size
        x_win = x.reshape(B * N, windows, self.window_size, D)

        qkv = (
            self.qkv(x_win)
            .reshape(
                B * N, windows, self.window_size, 3, self.num_heads, D // self.num_heads
            )
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(2, 3).reshape(B * N, windows, self.window_size, D)

        out = out.reshape(B * N, -1, D)
        if pad_len > 0:
            out = out[:, :T, :]
        if self.shift_size > 0:
            out = torch.roll(out, shifts=self.shift_size, dims=1)

        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        return self.proj(out)

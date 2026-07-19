import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Cross-modal fusion modules for M3RL.

The implementation fuses metric, log, and trace embeddings before STNet. Logs
serve as the query stream and metric/trace embeddings provide context, matching
the current code path used by both stages.
"""


class FlashCrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention block used by the fusion layer.

    The implementation uses PyTorch scaled dot-product attention and keeps the
    same interface for temporal and spatial fusion paths.
    """

    def __init__(
        self, d_model: int, num_heads: int, causal: bool = False, dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.n_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args
          q: (B, Lq, d_model) query sequence.
          k/v: (B, Lk, d_model) context sequences.
          attn_mask: optional attention mask.
        Returns
          out: (B, Lq, d_model)
        """
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_proj(out)
        out = self.drop(out)

        return q.transpose(1, 2).contiguous().view(B, Lq, self.d_model) + out


class CrossModalFusionLayer(nn.Module):
    """
    Cross-modal metric-log-trace fusion layer.

    Logs are used as the query modality, while metrics and traces provide
    context. The layer attends along both time and service dimensions, matching
    the paper's goal of capturing cross-modal interactions beyond static
    concatenation.
    """

    def __init__(self, settings, device):
        super(CrossModalFusionLayer, self).__init__()
        self.settings = settings
        self.device = device
        self.emb_dim = settings.emb_dim
        self.d_model = settings.d_model
        self.num_heads = settings.num_heads
        self.dropout = settings.dropout
        self.d_ff = settings.hidden_dim
        self.num_layers = 1

        kv_input_dim = self.emb_dim * 2

        self.key_proj = nn.Linear(kv_input_dim, self.d_model).to(device)
        self.value_proj = nn.Linear(kv_input_dim, self.d_model).to(device)
        self.query_proj = nn.Linear(self.emb_dim, self.d_model).to(device)

        self.temporal_cross_attn = nn.ModuleList(
            [
                FlashCrossAttentionBlock(self.d_model, self.num_heads)
                for _ in range(self.num_layers)
            ]
        )

        self.spatial_cross_attn = nn.ModuleList(
            [
                FlashCrossAttentionBlock(self.d_model, self.num_heads)
                for _ in range(self.num_layers)
            ]
        )

        fusion_hidden_dim = self.d_model * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, fusion_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(fusion_hidden_dim, self.d_model),
        ).to(device)
        self.fusion_norm = nn.LayerNorm(self.d_model).to(device)
        self.fusion_dropout = nn.Dropout(self.dropout)

    def forward(
        self, metric_emb: torch.Tensor, log_emb: torch.Tensor, trace_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            metric_emb (torch.Tensor): Metric embeddings (B, T, N, D').
            log_emb (torch.Tensor): Log embeddings (B, T, N, D'). Query source.
            trace_emb (torch.Tensor): Trace embeddings (B, T, N, D').

        Returns:
            torch.Tensor: Fused representation after dual cross-attention (B, T, N, D').
        """
        B, T, N, D_prime = log_emb.shape
        BN = B * N
        BT = B * T
        device = log_emb.device

        query_source = log_emb
        kv_source = torch.cat([metric_emb, trace_emb], dim=-1)

        q = self.query_proj(query_source)
        k = self.key_proj(kv_source)
        v = self.value_proj(kv_source)

        q_t = q.permute(0, 2, 1, 3).reshape(BN, T, -1)
        k_t = k.permute(0, 2, 1, 3).reshape(BN, T, -1)
        v_t = v.permute(0, 2, 1, 3).reshape(BN, T, -1)
        attn_output_t = q_t
        for layer in self.temporal_cross_attn:
            attn_output_t = layer(attn_output_t, k_t, v_t)
        fused_repr_temporal = attn_output_t.reshape(B, N, T, -1).permute(0, 2, 1, 3)

        q_s = q.reshape(BT, N, -1)
        k_s = k.reshape(BT, N, -1)
        v_s = v.reshape(BT, N, -1)
        attn_output_s = q_s
        for layer in self.spatial_cross_attn:
            attn_output_s = layer(attn_output_s, k_s, v_s)
        fused_repr_spatial = attn_output_s.reshape(B, T, N, -1)

        concatenated_repr = torch.cat([fused_repr_temporal, fused_repr_spatial], dim=-1)
        concatenated_flat = concatenated_repr.reshape(B * T * N, -1)
        fused_flat = self.fusion_mlp(concatenated_flat)
        final_fused_repr = fused_flat.reshape(B, T, N, -1)

        final_fused_repr = self.fusion_norm(final_fused_repr)
        final_fused_repr = self.fusion_dropout(final_fused_repr)

        return final_fused_repr

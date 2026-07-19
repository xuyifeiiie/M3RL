import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_softmax


class SparseMultiHeadGATv2(nn.Module):
    """
    Sparse multi-head graph attention for dynamic service dependencies.

    The layer consumes a batch of time-varying adjacency matrices and performs
    message passing only on observed dependency edges, which matches the M3RL
    paper's dynamic dependency modeling while avoiding dense all-pairs service
    attention.
    """

    def __init__(
        self,
        settings,
        device,
        input_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Parameter(torch.empty(heads, input_dim, out_dim))
        self.a = nn.Parameter(torch.empty(1, heads, 2 * out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W, gain=gain)
        nn.init.xavier_uniform_(self.a, gain=gain)

    def forward(self, x: torch.Tensor, adj_mx: torch.Tensor):
        """
        Args:
            x: Node representations with shape `(B, T, N, D_in)`.
            adj_mx: Dynamic dependency graph with shape `(B, T, N, N)`.

        Returns:
            Tensor with shape `(B, T, N, heads * out_dim)` when `concat=True`,
            otherwise `(B, T, N, out_dim)`.
        """
        B, T, N, _ = x.shape

        x_reshaped = x.reshape(B * T, N, -1)
        adj_reshaped = adj_mx.reshape(B * T, N, N)

        edge_index, edge_batch_idx = self.dense_to_sparse_batched(adj_reshaped)
        src_node, dst_node = edge_index[0], edge_index[1]

        h = torch.einsum("bni,hid->bnhd", x_reshaped, self.W)
        h_src = h[edge_batch_idx, src_node]
        h_dst = h[edge_batch_idx, dst_node]

        attention_input = torch.cat([h_src, h_dst], dim=-1)
        attention_score = torch.einsum("ehd,zhd->eh", attention_input, self.a)
        attention_score = self.leaky_relu(attention_score)

        alpha = scatter_softmax(attention_score, dst_node, dim=0, dim_size=N * B * T)
        alpha = self.dropout(alpha)

        messages = h_src * alpha.unsqueeze(-1)
        global_dst_node_idx = edge_batch_idx * N + dst_node
        out = scatter_add(messages, global_dst_node_idx, dim=0, dim_size=B * T * N)
        out = out.view(B * T, N, self.heads, self.out_dim)

        if self.concat:
            out = out.reshape(B * T, N, self.heads * self.out_dim)
        else:
            out = out.mean(dim=2)

        final_dim = out.shape[-1]
        return out.view(B, T, N, final_dim)

    @staticmethod
    def dense_to_sparse_batched(adj_batch: torch.Tensor):
        """Convert batched dense adjacency matrices to edge indices."""
        nonzero_coords = adj_batch.nonzero().t()
        batch_idx = nonzero_coords[0]
        src_nodes = nonzero_coords[1]
        dst_nodes = nonzero_coords[2]
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        return edge_index, batch_idx

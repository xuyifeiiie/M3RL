import torch
import torch.nn as nn
from models.M3RL.spatial import SparseMultiHeadGATv2
from models.M3RL.temporal import TemporalSelfAttentionLayer


class STNet(nn.Module):
    """
    Spatio-temporal backbone for M3RL.

    STNet alternates temporal self-attention with sparse graph attention. This
    implements the paper's dynamic dependency modeling: temporal blocks capture
    state evolution within each service, while graph attention propagates
    information along the time-varying service dependency matrix.
    """

    def __init__(self, settings, device, num_layers):
        super(STNet, self).__init__()
        self.settings = settings
        self.device = device
        self.d_model = settings.d_model
        self.num_st_layers = num_layers
        self.num_heads = settings.num_heads

        self.temporal_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()
        current_dim = self.d_model
        for i in range(self.num_st_layers):
            self.temporal_layers.append(TemporalSelfAttentionLayer(settings, device))
            gat_output_dim = self.d_model // self.num_heads
            self.spatial_layers.append(
                SparseMultiHeadGATv2(
                    settings, device, current_dim, gat_output_dim, heads=self.num_heads
                )
            )

        self.shortcut_conv = nn.Conv2d(self.d_model, self.d_model, kernel_size=(1, 1))

    def forward(
        self, fuse_input: torch.Tensor, dynamic_adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the STNet.

        Args:
            fuse_input (torch.Tensor): Fused multi-modal tensor, `(B, T, N, D)`.
            dynamic_adj (torch.Tensor): Dynamic service graph, `(B, T, N, N)`.

        Returns:
            torch.Tensor: Spatio-temporal representation, `(B, T, N, D)`.
        """

        x = fuse_input
        for i in range(self.num_st_layers):
            x = self.temporal_layers[i](x)
            x = self.spatial_layers[i](x, dynamic_adj)

        final_repr_btnd = x + self.shortcut_conv(
            fuse_input.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)

        return final_repr_btnd

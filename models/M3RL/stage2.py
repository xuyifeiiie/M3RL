import torch
import torch.nn as nn
from typing import Dict

from models.M3RL.stage1 import CrossModalRestorationBlock
from models.M3RL.stnet import STNet
from models.M3RL.common import (
    DeaggregationModuleTransformer,
    TemporalAttentionPooling,
)
from models.M3RL.embedding import EmbeddingLayer
from models.M3RL.fusion import CrossModalFusionLayer
from models.M3RL.common import SpatialAttentionPooling
from layers.forecast import QueryBasedForecastingHead


"""
Stage 2 implementation for M3RL.

This file contains the downstream fine-tuning model used for forecasting,
anomaly detection, and root cause localization. It reuses Stage 1 modules where
the code actually does so, then attaches task-specific heads. The forward path
is intentionally documented according to the current implementation rather than
being forced to match only the high-level paper description.
"""


class ResidualAdapter(nn.Module):
    """Residual bottleneck adapter used to refine localization logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = None,
        num_layers: int = 3,
        activation: nn.Module = nn.SiLU,
        use_layernorm: bool = True,
    ):
        """
        Args:
            input_dim (int): Dimension of the input tensor.
            hidden_dim (int): Dimension of the bottleneck/hidden layers.
            num_layers (int): Total number of linear layers in the adapter (min 2).
            output_dim (int, optional): Dimension of the output tensor.
            activation (nn.Module, optional): Activation function.
            use_layernorm (bool, optional): Whether to use LayerNorm.
        """
        super().__init__()

        if num_layers < 2:
            raise ValueError(
                "num_layers must be at least 2 for a bottleneck structure."
            )

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            layers.append(activation())
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(activation())
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))

        self.up_proj = nn.Linear(hidden_dim, output_dim)

        self.mlp_body = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the final projection layer to zero,
        making the adapter an identity-like transformation at the beginning.
        """
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        """
        if self.input_dim != self.output_dim:
            return self.up_proj(self.mlp_body(x))

        residual = self.up_proj(self.mlp_body(x))
        return x + residual


class MLPHead(nn.Module):
    """Generic MLP prediction head for downstream classification outputs."""

    def __init__(self, in_dim, out_dim, linear_sizes):
        super(MLPHead, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i - 1]
            layers += [nn.Linear(input_size, hidden), nn.SiLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):  # [batch_size, in_dim]
        return self.net(x)


class RootCauseLocalizationHead(nn.Module):
    """
    Node-level classifier for root cause localization.

    The head first pools each service over time with learned attention, then
    models interactions among service-level representations with a Transformer
    encoder before producing one logit per service.
    """

    def __init__(self, d_fuse, num_nodes, n_heads, num_layers):
        super().__init__()
        self.d_fuse = d_fuse
        self.temporal_attention_pool = nn.Sequential(
            nn.Linear(d_fuse, 2 * d_fuse), nn.GELU(), nn.Linear(2 * d_fuse, 1)
        )
        self.node_interaction_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_fuse,
                nhead=n_heads,
                dim_feedforward=d_fuse * 4,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(d_fuse, 1)

    def forward(self, fuse_out_btnd):
        node_representations = self.aggregate_time_with_attention(fuse_out_btnd)

        interacted_node_repr = self.node_interaction_transformer(node_representations)

        logits = self.classifier(interacted_node_repr)

        return logits.squeeze(-1)

    def aggregate_time_with_attention(self, x):
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_fuse)
        attn_weights = self.temporal_attention_pool(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weight_repr = torch.bmm(attn_weights.transpose(1, 2), x).squeeze(
            1
        )  # B * N, D_fuse
        node_representations = weight_repr.view(B, N, self.d_fuse)
        return node_representations


class M3RLStage2(nn.Module):
    """
    Stage 2: downstream fine-tuning model.

    The class reuses the Stage 1 embedding, cross-modal restoration, fusion,
    STNet, and aggregation/deaggregation modules, then attaches task-specific
    heads:

    - `detect`: system-level anomaly classification from the system vector;
    - `locate`: service-level root cause logits from decoded node embeddings;
    - `forecast`: future metric prediction from decoded STNet representations.

    Parameter names are intentionally kept compatible with Stage 1 so
    `load_pretrained_weights` can copy matching pretrained state_dict entries.
    """

    def __init__(self, settings, device):
        super(M3RLStage2, self).__init__()
        self.settings = settings
        self.device = device
        self.T = settings.history_steps
        self.N = settings.num_nodes
        self.d_model = settings.d_model
        self.d_sys = settings.d_model
        self.d_fuse = settings.d_model
        self.fusion_dim = settings.d_model
        self.hidden_dim = settings.hidden_dim
        self.history_steps = settings.history_steps
        self.freeze_pretrained = settings.freeze_pretrained
        self.num_nodes = settings.num_nodes
        self.num_heads = settings.num_heads
        self.dropout = settings.dropout
        self.num_features = settings.num_features

        self.n_samples = settings.n_samples
        self.emb_dim = settings.emb_dim
        modality_dims = {
            "metric": self.emb_dim,
            "log": self.emb_dim,
            "trace": self.emb_dim,
        }
        self.modalities = list(modality_dims.keys())

        self.embedding = EmbeddingLayer(settings, device).to(device)

        self.input_projections = nn.ModuleDict(
            {mod: nn.Linear(dim, self.d_model) for mod, dim in modality_dims.items()}
        )

        context_input_dim = self.d_model * 2
        self.context_k_projections = nn.ModuleDict(
            {mod: nn.Linear(context_input_dim, self.d_model) for mod in self.modalities}
        )
        self.context_v_projections = nn.ModuleDict(
            {mod: nn.Linear(context_input_dim, self.d_model) for mod in self.modalities}
        )

        self.d_ff = settings.hidden_dim or self.d_model * 4
        self.restoration_blocks = nn.ModuleDict(
            {
                mod: CrossModalRestorationBlock(
                    self.d_model, self.num_heads, self.d_ff, self.dropout
                )
                for mod in self.modalities
            }
        )

        self.output_projections = nn.ModuleDict(
            {mod: nn.Linear(self.d_model, dim) for mod, dim in modality_dims.items()}
        )

        self.fusion = CrossModalFusionLayer(settings, device).to(device)
        self.encoder_stnet = STNet(
            settings, device, num_layers=settings.num_encoder_layers
        )
        self.encoder_aggregation = TemporalAttentionPooling(input_dim=self.d_model)

        self.decoder_deaggregation = DeaggregationModuleTransformer(
            d_model=self.d_model, n_heads=self.num_heads, num_layers=3, timesteps=self.T
        )
        self.decoder_stnet = STNet(
            settings, device, num_layers=settings.num_decoder_layers
        ).to(device)

        self.localization_head = RootCauseLocalizationHead(
            self.d_fuse, self.num_nodes, self.num_heads, 6
        )

        self.detector = MLPHead(self.d_sys, 2, settings.detect_hiddens).to(device)
        self.localizer = MLPHead(
            self.num_nodes, self.num_nodes, settings.locate_hiddens
        ).to(device)

        self.spatial_projection = nn.Sequential(
            nn.Linear(self.num_nodes, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.history_steps, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.temporal_attention_pool = nn.Sequential(
            nn.Linear(self.d_fuse, 512), nn.Tanh(), nn.Linear(512, 1)
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.d_fuse, nhead=self.num_heads, batch_first=True, norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=3)

        self.adapter = ResidualAdapter(
            self.num_nodes, self.hidden_dim * 2, self.num_nodes, num_layers=6
        ).to(device)

        self.spatial_attention_pool = SpatialAttentionPooling(input_dim=self.d_model)

        self.layer_norm1 = nn.LayerNorm(self.emb_dim)
        self.layer_norm2 = nn.LayerNorm(self.d_sys)

        self.prediction_projection = nn.Sequential(
            nn.Linear(self.d_fuse, self.d_fuse * 2),
            nn.GELU(),
            nn.Linear(self.d_fuse * 2, self.d_fuse * 2),
            nn.GELU(),
            nn.Linear(self.d_fuse * 2, self.d_fuse * 2),
            nn.GELU(),
            nn.Linear(self.d_fuse * 2, self.d_fuse * 2),
            nn.GELU(),
            nn.Linear(self.d_fuse * 2, settings.num_features),
        )

        self.forecasthead = QueryBasedForecastingHead(
            d_model=self.d_fuse,
            n_heads=self.num_heads,
            T_future=self.T,
            N=self.num_nodes,
            C_metric=self.num_features,
        )

        if self.freeze_pretrained:
            self.freeze_layers()
        else:
            self.unfreeze_layers()

    def freeze_layers(self):
        print("Freezing pretrained layers.")
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.fusion.parameters():
            param.requires_grad = False
        for param in self.encoder_stnet.parameters():
            param.requires_grad = False
        for param in self.encoder_aggregation.parameters():
            param.requires_grad = False
        for param in self.input_projections.parameters():
            param.requires_grad = False
        for param in self.context_k_projections.parameters():
            param.requires_grad = False
        for param in self.context_v_projections.parameters():
            param.requires_grad = False
        for param in self.restoration_blocks.parameters():
            param.requires_grad = False
        for param in self.output_projections.parameters():
            param.requires_grad = False

    def unfreeze_layers(self):
        print("Unfreezing pretrained layers.")
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.fusion.parameters():
            param.requires_grad = True
        for param in self.encoder_stnet.parameters():
            param.requires_grad = True
        for param in self.encoder_aggregation.parameters():
            param.requires_grad = True
        for param in self.input_projections.parameters():
            param.requires_grad = False
        for param in self.context_k_projections.parameters():
            param.requires_grad = False
        for param in self.context_v_projections.parameters():
            param.requires_grad = False
        for param in self.restoration_blocks.parameters():
            param.requires_grad = False
        for param in self.output_projections.parameters():
            param.requires_grad = False

    def forward(
        self,
        metrics: torch.Tensor,
        logs: torch.Tensor,
        traces: torch.Tensor,
        adj_x: torch.Tensor,
        task: str = "detect",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for downstream tasks.

        Args:
            metrics, logs, traces: Multi-modal telemetry tensors with shape
                `(B, T, N, C_modality)`.
            adj_x: Dynamic dependency graph with shape `(B, T, N, N)`.
            task: One of `detect`, `locate`, or `forecast`.

        Returns:
            Task-specific outputs and the system representation.
        """
        metric_emb_orig, log_emb_orig, trace_emb_orig = self.embedding(
            metrics, logs, traces
        )
        B, T, N, D = metric_emb_orig.shape

        embs_dict_orig = {
            "metric": metric_emb_orig,
            "log": log_emb_orig,
            "trace": trace_emb_orig,
        }
        self.embs_dict = embs_dict_orig

        embs_proj = {
            mod: self.input_projections[mod](embs_dict_orig[mod])
            for mod in self.modalities
        }

        reconstructions_d_model = {}

        for query_mod in self.modalities:
            # The downstream model keeps the Stage 1 restoration path before fusion.
            query_emb = embs_proj[query_mod]

            context_mods = [m for m in self.modalities if m != query_mod]
            context_embs_list = [embs_proj[m] for m in context_mods]
            context_emb_cat = torch.cat(context_embs_list, dim=-1)

            key_emb = self.context_k_projections[query_mod](context_emb_cat)
            value_emb = self.context_v_projections[query_mod](context_emb_cat)

            restored_emb = self.restoration_blocks[query_mod](
                query_emb, key_emb, value_emb
            )
            reconstructions_d_model[query_mod] = restored_emb

        reconstructions = {}
        for mod in self.modalities:
            reconstructions[mod] = self.output_projections[mod](
                reconstructions_d_model[mod]
            )

        metric_emb = reconstructions["metric"]
        log_emb = reconstructions["log"]
        trace_emb = reconstructions["trace"]

        fuse_in = self.fusion(metric_emb, log_emb, trace_emb)

        # STNet produces node-time features and a pooled system representation.
        encoded_stnet_out = self.encoder_stnet(fuse_in, adj_x)
        system_vector = self.encoder_aggregation(encoded_stnet_out)

        if task == "detect":
            output = {}
            output["system_vector"] = system_vector
            output["detect_logits"] = self.detector(system_vector)
            output["locate_logits"] = None
        elif task == "locate":
            output = {}
            output["system_vector"] = system_vector
            deaggregated = self.decoder_deaggregation(system_vector)
            fuse_out_btnd = self.decoder_stnet(deaggregated, adj_x)

            # Root cause localization returns one score per service/node.
            node_representations = self.localization_head(fuse_out_btnd)
            node_representations = self.adapter(node_representations)

            system_vector_bd = self.spatial_attention_pool(system_vector)
            output["detect_logits"] = self.detector(system_vector_bd)
            output["locate_logits"] = node_representations
        elif task == "forecast":
            deaggregated = self.decoder_deaggregation(system_vector)
            fuse_out_btnd = self.decoder_stnet(deaggregated, adj_x)

            # Forecasting keeps multiple samples for the existing evaluation path.
            fuse_out_btnd = (
                fuse_out_btnd.unsqueeze(dim=1)
                .repeat(1, self.n_samples, 1, 1, 1)
                .reshape(-1, T, N, self.d_model)
            )
            nB, T, N, D_fuse = fuse_out_btnd.shape

            fuse_out_flat = fuse_out_btnd.reshape(nB * T * N, -1)
            metric_pred_flat = self.prediction_projection(fuse_out_flat)
            metric_pred = metric_pred_flat.reshape(nB, T, N, -1)
            metric_pred = metric_pred.reshape(B, self.n_samples, T, N, -1)
            output = metric_pred
        else:
            raise ValueError(f"Unknown task: {task}")

        return output, system_vector

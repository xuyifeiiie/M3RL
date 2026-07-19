import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from models.M3RL.stnet import STNet
from models.M3RL.embedding import EmbeddingLayer
from models.M3RL.fusion import CrossModalFusionLayer
from models.M3RL.contrast import CausalContrastiveLearning
from models.M3RL.common import TemporalAttentionPooling, DeaggregationModuleTransformer
from layers.forecast import QueryBasedForecastingHead


"""
Stage 1 implementation for M3RL.

This file contains the pre-training model used before downstream fine-tuning.
The implementation follows the code path used by the experiments: masked
multi-modal inputs are embedded, restored with cross-modal context, fused,
encoded by STNet, and optimized with restoration, future prediction, and causal
contrastive losses. The details here intentionally reflect the code behavior
even when a detail is not explicitly described in the paper text.
"""


class CrossModalRestorationBlock(nn.Module):
    """
    Restores one modality with context from the other two modalities.

    The block first attends across time for each service, then across services
    for each time step, mirroring the temporal and dependency-aware restoration
    used in the pre-training objective.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()

        self.temporal_cross_attn = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, "gelu", batch_first=True, norm_first=True
        )
        self.spatial_cross_attn = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, "gelu", batch_first=True, norm_first=True
        )

    def _cross_attention_forward(self, layer, query, key, value):
        q_norm = layer.norm1(query)
        k_norm = layer.norm1(key)
        v_norm = layer.norm1(value)
        attn_output, _ = layer.self_attn(q_norm, k_norm, v_norm, need_weights=False)
        x = query + layer.dropout1(attn_output)

        x_norm = layer.norm2(x)
        ff_output = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(x_norm)))
        )
        x = x + layer.dropout2(ff_output)
        return x

    def forward(self, query_emb, context_emb_k, context_emb_v):
        B, T, N, _ = query_emb.shape

        q_t = query_emb.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        k_t = context_emb_k.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        v_t = context_emb_v.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        temporal_out = self._cross_attention_forward(
            self.temporal_cross_attn, q_t, k_t, v_t
        )
        q_src_updated = temporal_out.view(B, N, T, -1).permute(0, 2, 1, 3)

        q_s = q_src_updated.reshape(B * T, N, -1)
        k_s = context_emb_k.reshape(B * T, N, -1)
        v_s = context_emb_v.reshape(B * T, N, -1)
        spatial_out = self._cross_attention_forward(
            self.spatial_cross_attn, q_s, k_s, v_s
        )

        restored_emb = spatial_out.view(B, T, N, -1)
        return restored_emb


class MetricPredictionHead(nn.Module):
    """Small MLP head for future metric prediction during Stage 1."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class M3RLStage1(nn.Module):
    """
    Stage 1: M3RL pre-training model.

    The model receives original telemetry and masked telemetry in parallel. It
    uses the original embeddings as reconstruction targets, restores each
    modality from the other modalities, fuses the restored metrics/logs/traces,
    and passes the fused representation through STNet over the dynamic
    dependency graph. The resulting representation is trained with:

    - cross-modal restoration loss over metric/log/trace embeddings;
    - future metric prediction loss from the decoded STNet representation;
    - causal contrastive loss for anomaly-aware representation learning.

    The attribute names are kept stable because checkpoint loading in Stage 2
    depends on these state_dict prefixes.
    """

    def __init__(self, settings, device):
        super().__init__()
        emb_dim = settings.emb_dim
        d_model = settings.d_model
        n_heads = settings.num_heads
        d_ff = settings.hidden_dim or d_model * 4
        dropout = settings.dropout
        self.num_nodes = settings.num_nodes
        self.num_features = settings.num_features

        self.modalities = ["metric", "log", "trace"]
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.num_heads = settings.num_heads
        self.T = settings.history_steps
        self.N = self.num_nodes
        self.freeze_pretrained = settings.freeze_pretrained

        self.embedding = EmbeddingLayer(settings, device).to(device)

        self.fusion_dim = self.d_model
        self.fusion = CrossModalFusionLayer(settings, device)

        self.encoder_stnet = STNet(
            settings, device, num_layers=settings.num_encoder_layers
        )
        self.encoder_aggregation = TemporalAttentionPooling(input_dim=self.d_model)

        self.aggregation_layer = TemporalAttentionPooling(input_dim=self.d_model)

        self.decoder_deaggregation = DeaggregationModuleTransformer(
            d_model=self.d_model, n_heads=self.num_heads, num_layers=3, timesteps=self.T
        )
        self.decoder_stnet = STNet(
            settings, device, num_layers=settings.num_decoder_layers
        )

        self.input_projections = nn.ModuleDict(
            {mod: nn.Linear(emb_dim, d_model) for mod in self.modalities}
        )

        context_input_dim = d_model * 2
        self.context_k_projections = nn.ModuleDict(
            {mod: nn.Linear(context_input_dim, d_model) for mod in self.modalities}
        )
        self.context_v_projections = nn.ModuleDict(
            {mod: nn.Linear(context_input_dim, d_model) for mod in self.modalities}
        )

        self.restoration_blocks = nn.ModuleDict(
            {
                mod: CrossModalRestorationBlock(d_model, n_heads, d_ff, dropout)
                for mod in self.modalities
            }
        )

        self.output_projections = nn.ModuleDict(
            {mod: nn.Linear(d_model, emb_dim) for mod in self.modalities}
        )

        self.vector_prediction_head = MetricPredictionHead(
            input_dim=d_model, output_dim=self.num_features, hidden_dim=4 * d_model
        )

        self.forecasthead = QueryBasedForecastingHead(
            d_model=self.d_model,
            n_heads=self.num_heads,
            T_future=self.T,
            N=self.num_nodes,
            C_metric=self.num_features,
        )

        self.causal_contrastive_learner = CausalContrastiveLearning(
            self.d_model, self.d_model, num_nodes=self.num_nodes, temperature=0.07
        )

        self.log_sigma_recon = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_predict = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_contrastive = nn.Parameter(torch.tensor(0.0))

        self.log_w = nn.Parameter(torch.zeros(3))

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
        for param in self.decoder_stnet.parameters():
            param.requires_grad = False
        for param in self.decoder_deaggregation.parameters():
            param.requires_grad = False

    def unfreeze_layers(self):
        print("Unfreezing pretrained layers.")
        for param in self.embedding.parameters():
            param.requires_grad = True
        for param in self.fusion.parameters():
            param.requires_grad = True
        for param in self.encoder_stnet.parameters():
            param.requires_grad = True
        for param in self.encoder_aggregation.parameters():
            param.requires_grad = True
        for param in self.decoder_stnet.parameters():
            param.requires_grad = True
        for param in self.decoder_deaggregation.parameters():
            param.requires_grad = True

    def forward(
        self,
        orig_inputs,
        mask_inputs: Tuple,
        dynamic_adj,
        future_data,
        sys_anomaly_labels,
        node_anomaly_labels,
    ) -> Tuple[Dict, Dict, None, None]:
        """
        Run the Stage 1 pre-training forward pass.

        Args:
            orig_inputs: Tuple of original metric/log/trace tensors.
            mask_inputs: Tuple of masked metric/log/trace tensors.
            dynamic_adj: Time-varying service dependency graph.
            future_data: Metric targets for the prediction objective.
            sys_anomaly_labels: System-level anomaly labels passed through for
                compatibility with the training interface.
            node_anomaly_labels: Node labels used by the causal contrastive
                objective.

        Returns:
            Reconstructed modality embeddings, original modality embeddings,
            and the restored system representation.
        """
        metric_emb_mask, log_emb_mask, trace_emb_mask = self.embedding(
            mask_inputs[0], mask_inputs[1], mask_inputs[2]
        )
        metric_emb_orig, log_emb_orig, trace_emb_orig = self.embedding(
            orig_inputs[0], orig_inputs[1], orig_inputs[2]
        )

        embs_dict_mask = {
            "metric": metric_emb_mask,
            "log": log_emb_mask,
            "trace": trace_emb_mask,
        }
        embs_dict_orig = {
            "metric": metric_emb_orig,
            "log": log_emb_orig,
            "trace": trace_emb_orig,
        }

        self.metric_future = future_data
        self.embs_dict_mask = embs_dict_mask
        self.embs_dict_orig = embs_dict_orig

        embs_proj = {
            mod: self.input_projections[mod](embs_dict_mask[mod])
            for mod in self.modalities
        }

        reconstructions_d_model = {}

        for query_mod in self.modalities:
            # Restore the current modality with the two remaining modalities as context.
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

        reconstructions_final = {}
        for mod in self.modalities:
            reconstructions_final[mod] = self.output_projections[mod](
                reconstructions_d_model[mod]
            )

        fuse_in_restored = self.fusion(
            reconstructions_final["metric"],
            reconstructions_final["log"],
            reconstructions_final["trace"],
        )

        # STNet encodes temporal evolution and dynamic service dependencies.
        encoded_stnet_out_restored = self.encoder_stnet(fuse_in_restored, dynamic_adj)
        self.system_vector_restored = self.encoder_aggregation(
            encoded_stnet_out_restored
        )

        # Decode the system vector back to node-time representations for prediction.
        deaggregated_restored = self.decoder_deaggregation(self.system_vector_restored)
        decode_stnet_out_restored = self.decoder_stnet(
            deaggregated_restored, dynamic_adj
        )
        self.metric_restored_predicted = self.forecasthead(decode_stnet_out_restored)

        node_embs = self.aggregation_layer(encoded_stnet_out_restored)

        self.causal_contrastive_loss = self.causal_contrastive_learner(
            node_embs_base=node_embs,
            node_anomaly_labels=node_anomaly_labels,
            emb_dict_base=reconstructions_final,
            dynamic_adj=dynamic_adj,
            model=self,
        )

        return (
            reconstructions_final,
            embs_dict_orig,
            self.system_vector_restored,
        )

    def loss_function(self, reconstructions: Dict, epoch) -> torch.Tensor:
        """
        Combine restoration, prediction, and contrastive pre-training losses.

        The schedule gradually shifts emphasis from denoising restoration toward
        future prediction while retaining the contrastive supervision.
        """
        recon_loss = 0.0
        for mod in self.modalities:
            recon_loss += F.mse_loss(reconstructions[mod], self.embs_dict_orig[mod])

        pred_loss = F.mse_loss(self.metric_restored_predicted, self.metric_future)

        causal_contrastive_loss = self.causal_contrastive_loss

        print(
            f"Reconstruction loss is {recon_loss}, prediction loss is {pred_loss}, causal contrastive loss is {causal_contrastive_loss}."
        )
        if epoch < 30:
            w1 = 0.2
            w2 = 0.6
            w3 = 1 - w1 - w2
        elif epoch < 60:
            w1 = 0.15
            w2 = 0.7
            w3 = 1 - w1 - w2
        else:
            w1 = 0.05
            w2 = 0.9
            w3 = 1 - w1 - w2

        return w1 * recon_loss + w2 * pred_loss + w3 * causal_contrastive_loss

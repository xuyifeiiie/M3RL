"""Causal contrastive objective used by M3RL Stage 1 pre-training."""

import random
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.M3RL.common import SpatialAttentionPooling


class CausalContrastiveLearning(nn.Module):
    """
    Contrast node-level representations with the root-cause service as positive.

    The module is the only contrastive component used by `M3RLStage1`. When a
    batch has no labeled abnormal node, it injects a synthetic single-node
    corruption into one modality and treats the corrupted node as the positive
    root cause.
    """

    def __init__(
        self,
        d_model_node: int,
        d_contrast: int,
        num_nodes: int,
        temperature: float = 0.1,
        noise_factor: float = 0.5,
    ):
        super().__init__()
        self.d_contrast = d_contrast
        self.num_nodes = num_nodes
        self.temperature = temperature
        self.noise_factor = noise_factor

        self.node_projection_head = nn.Linear(d_model_node, d_contrast)
        self.spatial_aggregator = SpatialAttentionPooling(d_contrast)

    def _inject_anomaly_on_restored(
        self, emb_dict_restored: Dict[str, torch.Tensor]
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Inject a single-node anomaly into one random modality per sample."""
        emb_dict_corr = {mod: emb.clone() for mod, emb in emb_dict_restored.items()}
        modalities = list(emb_dict_restored.keys())
        batch_size = next(iter(emb_dict_restored.values())).shape[0]
        device = next(iter(emb_dict_restored.values())).device

        corrupted_node_indices = torch.randint(
            0, self.num_nodes, (batch_size,), device=device
        )

        for batch_idx in range(batch_size):
            node_idx = corrupted_node_indices[batch_idx].item()
            modality = random.choice(modalities)
            target_embedding = emb_dict_corr[modality][batch_idx, :, node_idx, :]
            target_embedding += torch.randn_like(target_embedding) * self.noise_factor

        return emb_dict_corr, corrupted_node_indices

    def forward(
        self,
        node_embs_base: torch.Tensor,
        node_anomaly_labels: torch.Tensor,
        emb_dict_base: Dict[str, torch.Tensor],
        dynamic_adj: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        device = node_embs_base.device

        if node_anomaly_labels.sum() == 0:
            emb_dict_corr, root_cause_indices = self._inject_anomaly_on_restored(
                emb_dict_base
            )
            fused_corr = model.fusion(
                emb_dict_corr["metric"], emb_dict_corr["log"], emb_dict_corr["trace"]
            )
            key_node_embs = model.encoder_stnet(fused_corr, dynamic_adj).sum(dim=1)
        else:
            abnormal_sample_mask = node_anomaly_labels.sum(dim=1) > 0
            abnormal_sample_indices = torch.where(abnormal_sample_mask)[0]
            if abnormal_sample_indices.numel() == 0:
                return torch.tensor(0.0, device=device)
            key_node_embs = node_embs_base[abnormal_sample_indices]
            root_cause_indices = torch.argmax(
                node_anomaly_labels[abnormal_sample_indices], dim=1
            )

        projected_nodes = self.node_projection_head(key_node_embs)
        query_vectors = self.spatial_aggregator(projected_nodes)

        num_samples = query_vectors.shape[0]
        if num_samples == 0:
            return torch.tensor(0.0, device=device)

        query_vectors = F.normalize(query_vectors, p=2, dim=-1)
        projected_nodes = F.normalize(projected_nodes, p=2, dim=-1)

        gather_index = root_cause_indices.view(num_samples, 1, 1).expand(
            -1, -1, self.d_contrast
        )
        positive_keys = torch.gather(projected_nodes, 1, gather_index).squeeze(1)

        positive_logits = (query_vectors * positive_keys).sum(dim=-1, keepdim=True)
        negative_logits = torch.matmul(query_vectors, positive_keys.T)
        diagonal_mask = torch.eye(num_samples, dtype=torch.bool, device=device)
        negative_logits.masked_fill_(diagonal_mask, -float("inf"))

        logits = torch.cat([positive_logits, negative_logits], dim=1) / self.temperature
        labels = torch.zeros(num_samples, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

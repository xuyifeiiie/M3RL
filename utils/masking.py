import random
from typing import Tuple, Dict, Optional

import torch


class DynamicMaskingManager:
    """
    A class to apply various dynamic masking strategies to a batch of multi-modal data.
    All inputs and outputs are torch.Tensors.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the masking manager with a configuration dictionary.

        Example config:
        config = {
            "masking_prob": 0.8,  # Total probability of applying any mask to a sample
            "strategies": {
                "modality_drop": {"prob": 0.3},
                "spatiotemporal_point": {"prob": 0.3, "mask_ratio": 0.15},
                "spatiotemporal_block": {"prob": 0.4, "time_ratio_range": (0.1, 0.3), "node_ratio_range": (0.1, 0.3)}
            }
        }
        """
        self.config = self.get_default_config()
        if config:
            self.config.update(config)

    def get_default_config(self) -> Dict:
        """Provides a default configuration."""
        return {
            "masking_prob": 0.85,  # 80% chance to apply some form of mask, 20% chance to return original data
            "strategies": {
                # Probabilities below are relative, they will be normalized to sum to 1
                "modality_drop": {
                    "prob": 0.4,  # Relative probability of choosing this strategy
                    "num_modalities_to_drop": 1,  # How many modalities to drop if chosen
                    "drop_weights": [0.5, 0.25, 0.25],
                },
                "spatiotemporal_point": {
                    "prob": 0.3,
                    "mask_ratio": 0.15,  # Percentage of (T, N) points to mask
                },
                "spatiotemporal_block": {
                    "prob": 0.3,
                    "time_ratio_range": (
                        0.1,
                        0.4,
                    ),  # Min/max proportion of time steps in a block
                    "node_ratio_range": (
                        0.1,
                        0.4,
                    ),  # Min/max proportion of nodes in a block
                },
                # You can add feature-level masking here as well
            },
        }

    def _get_selected_strategy(self) -> str:
        """Randomly selects a masking strategy based on configured probabilities."""
        strategies = self.config["strategies"]
        names = list(strategies.keys())
        probs = [s["prob"] for s in strategies.values()]

        # Normalize probabilities
        total_prob = sum(probs)
        normalized_probs = [p / total_prob for p in probs]

        return random.choices(names, weights=normalized_probs, k=1)[0]

    def _apply_modality_drop(
        self, masks: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Applies modality-level masking based on a weight list provided in the config.
        """
        num_modalities = len(masks)
        cfg = self.config["strategies"]["modality_drop"]

        # Read modality-drop weights from the masking configuration.
        # The weight list must match the number of available modalities.
        drop_weights = cfg.get("drop_weights", [1.0] * num_modalities)
        num_to_drop = min(cfg["num_modalities_to_drop"], num_modalities - 1)

        if num_to_drop <= 0 or len(drop_weights) != num_modalities:
            if len(drop_weights) != num_modalities:
                print(
                    f"Warning: 'drop_weights' length ({len(drop_weights)}) does not match number of modalities ({num_modalities}). Using uniform sampling."
                )
                # Fallback to original uniform sampling
                indices_to_drop = random.sample(range(num_modalities), k=num_to_drop)
            else:
                return masks
        else:
            # Weighted sampling follows the paper's modality-corruption idea
            # without forcing all modalities to share the same drop frequency.
            indices_to_drop = random.choices(
                population=range(num_modalities), weights=drop_weights, k=num_to_drop
            )

        new_masks = list(masks)
        for idx in set(indices_to_drop):
            new_masks[idx].fill_(0.0)

        return tuple(new_masks)

    def _apply_spatiotemporal_point_mask(
        self, masks: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """Applies sparse point-wise masking to each modality independently."""
        cfg = self.config["strategies"]["spatiotemporal_point"]
        mask_ratio = cfg["mask_ratio"]

        new_masks = []
        for mask in masks:
            B, T, N, D = mask.shape
            num_elements_to_mask = int(B * T * N * mask_ratio)

            # Create a flat mask and shuffle it
            flat_mask = torch.cat(
                [
                    torch.zeros(num_elements_to_mask, dtype=mask.dtype),
                    torch.ones(B * T * N - num_elements_to_mask, dtype=mask.dtype),
                ]
            )
            shuffled_indices = torch.randperm(len(flat_mask))
            flat_mask = flat_mask[shuffled_indices]

            # Reshape and apply
            point_mask = flat_mask.view(B, T, N, 1).to(mask.device)
            new_masks.append(mask * point_mask)

        return tuple(new_masks)

    def _apply_spatiotemporal_block_mask(
        self, masks: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """Applies a continuous block mask. The same block is applied to all modalities."""
        cfg = self.config["strategies"]["spatiotemporal_block"]
        B, T, N, _ = masks[0].shape

        # Determine block size for T
        min_t_ratio, max_t_ratio = cfg["time_ratio_range"]
        block_t_size = int(T * random.uniform(min_t_ratio, max_t_ratio))
        block_t_size = max(1, block_t_size)  # ensure at least 1

        # Determine block size for N
        min_n_ratio, max_n_ratio = cfg["node_ratio_range"]
        block_n_size = int(N * random.uniform(min_n_ratio, max_n_ratio))
        block_n_size = max(1, block_n_size)

        # Determine block start position (for each sample in the batch)
        # We can apply different blocks to different samples in the batch for more variety
        start_t = torch.randint(0, T - block_t_size + 1, (B,)).tolist()
        start_n = torch.randint(0, N - block_n_size + 1, (B,)).tolist()

        new_masks = []
        for mask in masks:
            new_mask = mask.clone()
            for i in range(B):
                new_mask[
                    i,
                    start_t[i] : start_t[i] + block_t_size,
                    start_n[i] : start_n[i] + block_n_size,
                    :,
                ] = 0.0
            new_masks.append(new_mask)

        return tuple(new_masks)

    def forward(
        self, modality_tensors: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """
        The main method to generate masks and apply them.

        Args:
            modality_tensors (Tuple[torch.Tensor, ...]): A tuple of tensors, one for each modality,
                                                         each with shape (B, T, N, D_mod).

        Returns:
            Tuple[torch.Tensor, ...]: The tuple of masked tensors.
        """
        # Overall probability check
        if random.random() > self.config["masking_prob"]:
            # Return original tensors (chance for model to see clean data)
            return modality_tensors

        # Create initial all-ones masks
        masks = tuple(torch.ones_like(tensor) for tensor in modality_tensors)

        # Select and apply a strategy
        strategy = self._get_selected_strategy()

        if strategy == "modality_drop":
            masked_masks = self._apply_modality_drop(masks)
        elif strategy == "spatiotemporal_point":
            masked_masks = self._apply_spatiotemporal_point_mask(masks)
        elif strategy == "spatiotemporal_block":
            masked_masks = self._apply_spatiotemporal_block_mask(masks)
        else:
            masked_masks = masks  # Should not happen if config is correct

        # Apply the final generated masks to the input tensors via element-wise multiplication
        masked_tensors = tuple(
            tensor * mask for tensor, mask in zip(modality_tensors, masked_masks)
        )

        return masked_tensors

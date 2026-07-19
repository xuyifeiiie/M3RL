import torch
import torch.nn as nn
import random
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from copy import deepcopy


class PerturbationManager:
    def __init__(self, settings, device: torch.device):
        self.settings = settings
        self.device = device

    def _should_perturb_this_sample(self) -> bool:
        """Determines if a sample should be chosen for raw_data perturbation."""
        return random.random() < self.settings.raw_data_perturb_rate

    def _perturb_numeric_tensor_for_sample(
        self,
        sample_tensor_tnc: np.ndarray,  # (T, N, C) - Numpy array
        target_node_idx: int,
    ) -> np.ndarray:
        """Applies noise to a single sample's numeric tensor for a target node."""
        # Convert to tensor for PyTorch operations, then back to numpy
        temp_tensor = torch.from_numpy(sample_tensor_tnc).float().to(self.device)
        T_s, N_s, C_s = temp_tensor.shape
        perturbed_sample_tensor = temp_tensor.clone()
        node_data_to_perturb = perturbed_sample_tensor[
            :, target_node_idx, :
        ].clone()  # (T, C)

        noise_type = self.settings.metric_trace_noise_type
        std = self.settings.metric_trace_noise_std
        target_dims_mode = self.settings.metric_trace_target_dims
        num_dims_to_noise_cfg = min(self.settings.metric_trace_num_dims_to_noise, C_s)

        dims_to_noise = []
        if C_s > 0:  # Ensure there are channels to choose from
            if target_dims_mode == "all":
                dims_to_noise = list(range(C_s))
            elif target_dims_mode == "random":
                dims_to_noise = random.sample(
                    range(C_s), min(num_dims_to_noise_cfg, C_s)
                )

        for t_step in range(T_s):
            for c_dim in dims_to_noise:
                original_value = node_data_to_perturb[t_step, c_dim]
                if noise_type == "gaussian":
                    noise = torch.randn_like(original_value) * std
                elif noise_type == "uniform":
                    noise = (torch.rand_like(original_value) - 0.5) * 2 * std
                elif noise_type == "replace_random":
                    node_data_to_perturb[t_step, c_dim] = torch.randn_like(
                        original_value
                    )
                    continue
                else:
                    noise = 0.0
                node_data_to_perturb[t_step, c_dim] += noise
        perturbed_sample_tensor[:, target_node_idx, :] = node_data_to_perturb
        return perturbed_sample_tensor.cpu().numpy()

    def _apply_modality_drop_internal(
        self, data_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Applies modality drop. Input values are numpy arrays."""
        perturbed_data = data_dict  # Modify in place or copy if needed by caller

        # [Logic Fix]: Handle the list correctly. Avoid creating [[...]] nested lists.
        drop_list = self.settings.modality_drop_list
        if not isinstance(drop_list, list):
            drop_list = [drop_list]

        for modality_name in drop_list:
            target_key = modality_name

            # Map configuration names to tensor keys in the batch.
            if modality_name == "logs":
                # Dropping logs masks both semantic embeddings and log-count features.
                keys_to_drop = ["log_emb", "log_stats"]
            else:
                keys_to_drop = [modality_name]

            for key in keys_to_drop:
                if key in perturbed_data and isinstance(
                    perturbed_data[key], np.ndarray
                ):
                    print(f"    Dropping modality by zeroing: {key}")
                    perturbed_data[key] = np.zeros_like(perturbed_data[key])

        return perturbed_data

    def _apply_spatiotemporal_mask_internal(
        self, data_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Applies spatiotemporal masks. Input values are numpy arrays."""
        perturbed_data = data_dict
        mask_type = self.settings.mask_type
        mask_val = float(self.settings.mask_value)

        # Get B, T, N from the first available tensor
        T, N = -1, -1
        for key, tensor_data in data_dict.items():
            if (
                isinstance(tensor_data, np.ndarray) and tensor_data.ndim == 3
            ):  # Expecting (T,N,C)
                T, N, _ = tensor_data.shape
                break
        if T == -1:
            return perturbed_data  # No suitable tensor found

        # [Logic Fix]: Pre-calculate indices OUTSIDE the modality loop to ensure consistency
        nodes_to_mask_indices = []
        times_to_mask_indices = []
        spatiotemporal_points = []

        if "spatial" in mask_type:
            scope_n = self.settings.mask_scope_nodes
            prop_n = self.settings.mask_node_proportion
            num_mask_n = (
                int(N * prop_n)
                if scope_n == "random"
                else self.settings.mask_node_num_fixed
            )
            if N > 0:
                nodes_to_mask_indices = random.sample(range(N), min(num_mask_n, N))

        if "temporal" in mask_type:
            scope_t = self.settings.mask_scope_time
            prop_t = self.settings.mask_time_proportion
            num_mask_t = (
                int(T * prop_t)
                if scope_t == "random"
                else self.settings.mask_time_num_fixed
            )
            if T > 0:
                times_to_mask_indices = random.sample(range(T), min(num_mask_t, T))

        if mask_type == "spatiotemporal":
            # [Logic Fix]: Generate random points once for all modalities
            prop_st = self.settings.mask_spatiotemporal_proportion
            num_st_points_to_mask_total = int(T * N * prop_st)
            for _ in range(num_st_points_to_mask_total):
                spatiotemporal_points.append(
                    (random.randint(0, T - 1), random.randint(0, N - 1))
                )

        target_modalities = ["metrics", "log_emb", "traces", "log_stats"]

        for modality_key in target_modalities:
            if (
                modality_key in perturbed_data
                and isinstance(perturbed_data[modality_key], np.ndarray)
                and perturbed_data[modality_key].ndim == 3
            ):

                tensor_to_mask = perturbed_data[modality_key]
                original_sum = np.sum(tensor_to_mask)

                # `log_stats` is handled by its own iteration over mapped keys.

                if mask_type == "spatial" and nodes_to_mask_indices:
                    tensor_to_mask[:, nodes_to_mask_indices, :] = mask_val
                    print(
                        f"    Spatial mask applied to {modality_key}, nodes: {nodes_to_mask_indices}"
                    )

                elif mask_type == "temporal" and times_to_mask_indices:
                    tensor_to_mask[times_to_mask_indices, :, :] = mask_val
                    print(
                        f"    Temporal mask applied to {modality_key}, timesteps: {times_to_mask_indices}"
                    )

                elif mask_type == "spatiotemporal" and spatiotemporal_points:
                    for r_t, r_n in spatiotemporal_points:
                        tensor_to_mask[r_t, r_n, :] = mask_val

                perturbed_data[modality_key] = tensor_to_mask

                # Optional runtime check that the perturbation modified the tensor.
                new_sum = np.sum(tensor_to_mask)
                if original_sum != 0 and new_sum == original_sum:
                    print(
                        f"⚠️ WARNING: {modality_key} was processed but sum did not change! (Maybe mask_val same as data?)"
                    )
                else:
                    # A non-zero difference confirms the perturbation was applied.
                    pass

        return perturbed_data

    def apply_perturbations_to_chunk(
        self, chunk_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Applies configured perturbations to a single chunk's data (after FeatureExtractor).
        Modifies 'metrics', 'log_emb', 'traces', and 'labels'.
        Assumes chunk_data['log_emb'] is present and other modalities are numpy arrays (T,N,C).
        chunk_data['labels'] is a scalar: -1 or node_index (0 to N-1).
        """
        if not self.settings.if_perturb:
            return chunk_data

        perturbed_chunk = deepcopy(chunk_data)  # Work on a copy

        original_label = perturbed_chunk.get("labels", -1)  # Get original label
        # Ensure labels is a numpy array for consistent modification
        if not isinstance(original_label, np.ndarray):
            original_label = np.array(original_label)

        perturb_type_setting = self.settings.perturb_type

        if perturb_type_setting == "raw_data":
            if self._should_perturb_this_sample():  # Probabilistic decision
                if (
                    original_label.item() == -1
                    or not self.settings.perturb_only_normal_samples
                ):
                    # Perturb this normal sample, or perturb abnormal samples too if config allows
                    is_originally_normal = original_label.item() == -1
                    target_node_for_perturb = random.randint(
                        0, self.settings.num_nodes - 1
                    )
                    print(
                        f"    Raw data perturbation: Targeting node {target_node_for_perturb} in chunk."
                    )

                    scope = self.settings.raw_data_scope
                    target_mod = self.settings.raw_data_target_modality

                    if perturbed_chunk.get("metrics") is not None and (
                        scope == "mix" or target_mod == "metrics"
                    ):
                        perturbed_chunk["metrics"] = (
                            self._perturb_numeric_tensor_for_sample(
                                perturbed_chunk["metrics"], target_node_for_perturb
                            )
                        )
                        if is_originally_normal:
                            perturbed_chunk["labels"] = np.array(
                                target_node_for_perturb
                            )
                        print(
                            f"      Metrics perturbed for node {target_node_for_perturb}."
                        )

                    if perturbed_chunk.get("traces") is not None and (
                        scope == "mix" or target_mod == "traces"
                    ):
                        perturbed_chunk["traces"] = (
                            self._perturb_numeric_tensor_for_sample(
                                perturbed_chunk["traces"], target_node_for_perturb
                            )
                        )
                        if is_originally_normal:
                            perturbed_chunk["labels"] = np.array(
                                target_node_for_perturb
                            )
                        print(
                            f"      Traces perturbed for node {target_node_for_perturb}."
                        )

        elif perturb_type_setting == "modality_drop":
            # Modality drop does not change labels for robustness testing
            perturbed_chunk = self._apply_modality_drop_internal(perturbed_chunk)

        elif perturb_type_setting == "spatiotemporal_mask":
            # Masking also typically does not change ground truth labels for robustness testing
            perturbed_chunk = self._apply_spatiotemporal_mask_internal(perturbed_chunk)

        return perturbed_chunk

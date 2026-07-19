"""
Convert legacy NPZ chunks to HDF5.

M3RL's current Eadro workflow reads processed HDF5 files directly. This script
is retained as a utility for converting older NPZ chunk dictionaries into HDF5
when reproducing experiments generated before the HDF5 pipeline was
standardized.
"""

import h5py
import numpy as np
import os
import sys
import json
import argparse

NODE_SEPARATOR = "\x1D"
KEY_VALUE_SEPARATOR = "\x1E"
LOG_SEPARATOR = "\x1F"

parser = argparse.ArgumentParser(description="arguments")
parser.add_argument("--dataset", type=str, default="SN-Eadro", help="Dataset name")
args = parser.parse_args()

current_root_path = os.getcwd()
if current_root_path.endswith("data"):
    pass
else:
    current_root_path = os.path.join(current_root_path, "data")
dataset_dir_name = args.dataset + " Dataset"

train_npz_file_path = os.path.join(
    current_root_path, dataset_dir_name, "processed", "train.npz"
)
valid_npz_file_path = os.path.join(
    current_root_path, dataset_dir_name, "processed", "val.npz"
)
test_npz_file_path = os.path.join(
    current_root_path, dataset_dir_name, "processed", "test.npz"
)

train_h5_file_path = os.path.join(
    current_root_path, dataset_dir_name, "processed", "train.h5"
)
valid_h5_file_path = os.path.join(
    current_root_path, dataset_dir_name, "processed", "val.h5"
)
test_h5_file_path = os.path.join(
    current_root_path, dataset_dir_name, "processed", "test.h5"
)

if __name__ == "__main__":
    for npz_file_path, h5_file_path in zip(
        [train_npz_file_path, valid_npz_file_path, test_npz_file_path],
        [train_h5_file_path, valid_h5_file_path, test_h5_file_path],
    ):
        print(f"Loading NPZ file: {npz_file_path} ...")
        try:
            loaded_npz = np.load(npz_file_path, allow_pickle=True)

            if "data" not in loaded_npz:
                print(f"Error: key 'data' was not found in '{npz_file_path}'.")
                exit()

            all_chunks_data = loaded_npz["data"].item()
            loaded_npz.close()
            print(f"Loaded {len(all_chunks_data)} chunks from the NPZ file.")

        except FileNotFoundError:
            print(f"Error: file '{npz_file_path}' was not found.")
            exit()
        except Exception as e:
            print(f"Error while loading NPZ file: {e}")
            exit()

        if not isinstance(all_chunks_data, dict):
            print(
                f"Error: the 'data' entry in the NPZ file is {type(all_chunks_data)}, not dict."
            )
            exit()
        if not all_chunks_data:
            print("Warning: the NPZ 'data' dictionary is empty.")

        # --- 2. Try to determine a global static adjacency matrix ---
        # If all chunks share the same static graph, storing one global copy
        # keeps the HDF5 file compact. The first chunk is used as a candidate.
        potential_global_static_adj = None
        first_chunk_id_for_static_adj = None

        if all_chunks_data:
            # Read the first chunk as the static-adjacency candidate.
            first_chunk_id_for_static_adj = next(iter(all_chunks_data))
            first_chunk_content = all_chunks_data[first_chunk_id_for_static_adj]

            if "static_adj" in first_chunk_content:
                try:
                    potential_global_static_adj = np.asarray(
                        first_chunk_content["static_adj"]
                    )
                    print(
                        f"Loaded 'static_adj' from chunk '{first_chunk_id_for_static_adj}' as the 'static_adj_global' candidate."
                    )
                except Exception as e:
                    print(
                        f"Warning: failed to convert 'static_adj' from chunk '{first_chunk_id_for_static_adj}' to a NumPy array: {e}"
                    )
                    potential_global_static_adj = None
            else:
                print(
                    f"Warning: chunk '{first_chunk_id_for_static_adj}' has no 'static_adj'; 'static_adj_global' will not be saved."
                )

        # --- 3. Write the HDF5 file ---
        print(f"Creating HDF5 file: {h5_file_path}...")
        try:
            with h5py.File(h5_file_path, "w") as hf:
                if potential_global_static_adj is not None:
                    try:
                        hf.create_dataset(
                            "static_adj_global", data=potential_global_static_adj
                        )
                        print("Saved 'static_adj_global' at the HDF5 root.")
                    except Exception as e:
                        print(f"Error while saving 'static_adj_global' to HDF5: {e}")

                num_chunks = len(all_chunks_data)
                for i, (chunk_id, chunk_content) in enumerate(all_chunks_data.items()):
                    if i % 100 == 0 or i == num_chunks - 1:
                        print(f"  Processing chunk {i+1}/{num_chunks}: ID '{chunk_id}'")

                    chunk_id_str = str(chunk_id)
                    try:
                        grp = hf.create_group(chunk_id_str)
                    except ValueError as ve:
                        print(
                            f"  Failed to create group '{chunk_id_str}': {ve}. Skipping this chunk."
                        )
                        continue
                    if not isinstance(chunk_content, dict):
                        print(
                            f"  Warning: chunk '{chunk_id_str}' is not a dictionary. Skipping."
                        )
                        continue

                    for key, value in chunk_content.items():
                        try:
                            if key == "logs":
                                if not isinstance(value, dict):
                                    print(
                                        f"    Warning: chunk '{chunk_id_str}' field 'logs' should be a dictionary (got {type(value)}). Logs are not saved for this field."
                                    )
                                    continue

                                logs_main_group = grp.create_group("logs")

                                for (
                                    t_step_key,
                                    nodes_dict_for_timestep,
                                ) in value.items():  # Numeric time-step key.
                                    if not isinstance(nodes_dict_for_timestep, dict):
                                        print(
                                            f"    Warning: chunk '{chunk_id_str}', logs, time '{t_step_key}' is not a dictionary. Skipping this time step."
                                        )
                                        continue

                                    encoded_nodes_for_timestep = []
                                    # Node order is encoded explicitly by node_id.
                                    for (
                                        node_id,
                                        node_log_list,
                                    ) in (
                                        nodes_dict_for_timestep.items()
                                    ):  # Numeric node identifier.
                                        if not isinstance(
                                            node_log_list, list
                                        ) or not all(
                                            isinstance(s, str) for s in node_log_list
                                        ):
                                            print(
                                                f"    Warning: chunk '{chunk_id_str}', logs, time '{t_step_key}', node '{node_id}' is not a list of strings. Skipping this node."
                                            )
                                            continue

                                        joined_node_logs = LOG_SEPARATOR.join(
                                            node_log_list
                                        )
                                        # Store node_id as text before joining with KEY_VALUE_SEPARATOR.
                                        encoded_node_data = (
                                            str(node_id)
                                            + KEY_VALUE_SEPARATOR
                                            + joined_node_logs
                                        )
                                        encoded_nodes_for_timestep.append(
                                            encoded_node_data
                                        )

                                    final_encoded_string_for_timestep = (
                                        NODE_SEPARATOR.join(encoded_nodes_for_timestep)
                                    )

                                    # HDF5 dataset names use the time-step key string.
                                    logs_main_group.create_dataset(
                                        str(t_step_key),
                                        data=final_encoded_string_for_timestep,
                                        dtype=h5py.string_dtype(encoding="utf-8"),
                                    )
                            else:
                                data_to_save = np.asarray(value)
                                grp.create_dataset(key, data=data_to_save)
                        except Exception as e_ds:
                            print(
                                f"    Error while saving field '{key}' for chunk '{chunk_id_str}': {e_ds}. Field skipped."
                            )
            print(f"\nHDF5 conversion finished: {h5_file_path}")
            if os.path.exists(h5_file_path):
                print(
                    f"HDF5 file size: {os.path.getsize(h5_file_path) / (1024**2):.2f} MB"
                )

        except Exception as e:
            print(f"Unexpected error while creating HDF5 or processing data: {e}")

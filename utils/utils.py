import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import h5py
import torch
import random
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.semantics import FeatureExtractor
from utils.perturb import PerturbationManager


def log_string(log, string):
    """Write a message to both the log file and stdout."""
    log.write(string + "\n")
    log.flush()
    print(string)


def scan_category(base_path):
    """
    Scan category directories from raw observation data
    :param base_path:
    :return: category list
    """
    _, base_folders = get_subfiles_subfolders(base_path)
    category_list = []
    for category_path in base_folders:
        if os.path.isdir(category_path):
            if sys.platform.startswith("win"):
                category = category_path.split("\\")[-1]
                if category.find("ipynb") != -1:
                    continue
                elif category.find("processed") != -1:
                    continue
                elif category.find("pycache") != -1:
                    continue
                else:
                    category_list.append(category)
            elif sys.platform.startswith("linux"):
                category = category_path.split("/")[-1]
                if category.find("ipynb") != -1:
                    continue
                elif category.find("processed") != -1:
                    continue
                elif category.find("pycache") != -1:
                    continue
                else:
                    category_list.append(category)

    return category_list


def get_subfiles_subfolders(folder_path):
    """
    Get subfile and subfolders of a directory
    :param folder_path: directory path
    :return: subfile and subfolders path list
    """
    files = []
    subfolders = []
    abs_path = ""

    if sys.platform.startswith("win"):
        abs_path = os.path.abspath(folder_path)
    elif sys.platform.startswith("linux"):
        abs_path = folder_path

    for file in os.listdir(abs_path):
        file_path = os.path.join(abs_path, file)
        if os.path.isdir(file_path):
            subfolders.append(file_path)
        else:
            files.append(file_path)
    return files, subfolders


def get_dirname(dir, index=-1):
    if sys.platform.startswith("win"):
        return dir.split("\\")[index]
    elif sys.platform.startswith("linux"):
        return dir.split("/")[index]
    else:
        sys.exit(0)


def save_prediction_result(samples, targets, predict_steps, save_path):
    samples = torch.cat(samples, dim=0)[:50]
    targets = torch.cat(targets, dim=0)[:50]
    observed_flag = torch.ones_like(targets)  # B T N F
    evaluate_flag = observed_flag  # B T N F
    evaluate_flag[:, -predict_steps:, :, :] = 1  # later T_p values set to 1
    with open(save_path, "wb") as f:
        pickle.dump([samples, targets, observed_flag, evaluate_flag], f)


def save_sv_result(sv_list, save_path):
    all_system_vectors = torch.cat(sv_list, dim=0)
    system_vectors_numpy = all_system_vectors.cpu().numpy()
    with open(save_path, "wb") as f:
        # Store the merged system-vector array as a single object.
        pickle.dump(system_vectors_numpy, f)

    print("Saving complete.")


def count_parameters(model):
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable nondeterministic cuDNN behavior to maximize reproducibility."""
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_adjacency_matrix(
    edge_filename, span_df, num_of_instances, type_="connectivity", id_filename=None
):
    """
    :param edge_filename: str, csv edge info file path
    :param num_of_instances:int, number of nodes
    :param type_:str, {connectivity, distance}
    :param id_filename:str node id: index
    """
    A = np.zeros((int(num_of_instances), int(num_of_instances)), dtype=np.float32)

    if id_filename:
        with open(id_filename, "r") as f:
            id_dict = {
                int(index): node_id
                for node_id, index in enumerate(f.read().strip().split("\n"))
            }
        df = pd.read_csv(edge_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            A[id_dict[i], id_dict[j]] = 1
            A[id_dict[j], id_dict[i]] = 1

        return A

    if edge_filename == "":
        df = span_df
    else:
        df = pd.read_csv(edge_filename)
    for row in df.values:
        if len(row) != 6:
            continue
        i, j, latency, counts = int(row[2]), int(row[3]), float(row[4]), float(row[5])
        if type_ == "connectivity":
            A[i, j] = 1
        elif type_ == "counts":
            A[i, j] = counts
        elif type_ == "count_div_latency":
            if latency == 0:
                A[i, j] = 0
            else:
                A[i, j] = counts / (latency / 1000000)
        else:
            raise ValueError("type_ error, must be " "connectivity or distance!")

    return A


def standard_transform(data: np.array) -> np.array:
    """standard normalization.

    Args:
        data (np.array): raw time series data.
    Returns:
        np.array: normalized raw time series data.
    """
    t, n, f = data.shape
    data = data.reshape(-1, f)
    mean, std = data.mean(axis=0), data.std(axis=0)
    scaler = {
        "func": standard_re_transform.__name__,
        "args": {"mean": mean, "std": std},
    }

    def normalize(x):
        return (x - mean) / std

    data_norm = normalize(data)
    data_norm = data_norm.reshape(t, n, f)
    return data_norm, scaler


def standard_re_transform(x, **kwargs):
    mean, std = kwargs["mean"], kwargs["std"]
    x = x * std
    x = x + mean
    return x


NODE_SEPARATOR = "\x1D"  # Separates serialized node payloads.
KEY_VALUE_SEPARATOR = "\x1E"  # Separates a node ID from its serialized logs.
LOG_SEPARATOR = "\x1F"  # Separates multiple log entries for one node.


# --- Helper PyTorch Dataset: reads individual chunks from raw HDF5 files ---
class HDF5RawChunkReaderDataset(Dataset):

    def __init__(self, h5_file_path, parse_log_keys_to_int=True):
        if not os.path.exists(h5_file_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_file_path}")
        self.h5_file_path = h5_file_path
        self.h5file = None
        self.global_static_adj = None
        self.chunk_ids = []
        self.parse_log_keys_to_int = parse_log_keys_to_int
        self._initialize_metadata()

    def _initialize_metadata(self):
        try:
            with h5py.File(self.h5_file_path, "r") as temp_h5file:
                global_keys = [
                    "static_adj_global"
                ]  # Assuming this might exist in raw HDF5
                for key_in_h5 in temp_h5file.keys():
                    if key_in_h5 not in global_keys and isinstance(
                        temp_h5file[key_in_h5], h5py.Group
                    ):
                        self.chunk_ids.append(key_in_h5)
                if not self.chunk_ids:
                    raise ValueError(
                        f"No chunk data (HDF5 groups) found in raw HDF5 file: {self.h5_file_path}"
                    )
                try:
                    self.chunk_ids.sort(key=lambda x: int(x) if x.isdigit() else str(x))
                except ValueError:
                    self.chunk_ids.sort()

                if "static_adj_global" in temp_h5file:
                    ds_global_adj = temp_h5file["static_adj_global"]
                    if ds_global_adj.shape is None or ds_global_adj.shape == ():
                        self.global_static_adj = ds_global_adj[()]
                    else:
                        self.global_static_adj = ds_global_adj[:]
        except Exception as e:
            raise IOError(f"Metadata init error for raw HDF5 {self.h5_file_path}: {e}")

    def _ensure_file_open(self):
        if self.h5file is None:
            try:
                self.h5file = h5py.File(self.h5_file_path, "r")
            except Exception as e:
                raise IOError(f"Cannot open raw HDF5 {self.h5_file_path}: {e}")

    def __len__(self):
        return len(self.chunk_ids)

    def _parse_key_to_int_if_needed(self, key_str):
        if self.parse_log_keys_to_int:
            try:
                return int(key_str)
            except ValueError:
                return key_str
        return key_str

    def _read_hdf5_field(self, grp, field_name):
        if field_name in grp:
            dataset = grp[field_name]
            if dataset.shape is None or dataset.shape == ():
                return dataset[()]
            else:
                return dataset[:]
        return None

    def __getitem__(self, idx):  # Reads RAW data, including text logs
        self._ensure_file_open()
        if not (0 <= idx < len(self.chunk_ids)):
            raise IndexError("Index out of bounds.")
        chunk_id_str = self.chunk_ids[idx]
        try:
            grp = self.h5file[chunk_id_str]
        except KeyError:
            raise KeyError(f"Chunk ID '{chunk_id_str}' not found in raw HDF5.")

        sample = {"idx": idx}

        reconstructed_logs_dict = {}
        if "logs" in grp and isinstance(grp["logs"], h5py.Group):
            logs_main_group = grp["logs"]
            sorted_timestep_keys_str = sorted(
                list(logs_main_group.keys()),
                key=(
                    self._parse_key_to_int_if_needed
                    if self.parse_log_keys_to_int
                    else None
                ),
            )
            for t_step_key_str in sorted_timestep_keys_str:
                encoded_str_bytes = self._read_hdf5_field(
                    logs_main_group, t_step_key_str
                )
                if encoded_str_bytes is None:
                    encoded_str_bytes = ""
                if isinstance(encoded_str_bytes, bytes):
                    encoded_str = encoded_str_bytes.decode("utf-8")
                elif isinstance(encoded_str_bytes, str):
                    encoded_str = encoded_str_bytes
                else:
                    encoded_str = ""
                current_nodes_dict = {}
                if encoded_str:
                    node_data_list = encoded_str.split(NODE_SEPARATOR)
                    for item in node_data_list:
                        if not item:
                            continue
                        parts = item.split(KEY_VALUE_SEPARATOR, 1)
                        if len(parts) == 2:
                            node_id_s, joined_logs_s = parts
                            node_id_p = self._parse_key_to_int_if_needed(node_id_s)
                            current_nodes_dict[node_id_p] = (
                                joined_logs_s.split(LOG_SEPARATOR)
                                if joined_logs_s
                                else []
                            )
                reconstructed_logs_dict[
                    self._parse_key_to_int_if_needed(t_step_key_str)
                ] = current_nodes_dict
        sample["logs"] = reconstructed_logs_dict if reconstructed_logs_dict else {}

        other_fields = [
            "log_stats",
            "metrics",
            "traces",
            "adj",
            "y_labels",
            "labels",
            "static_adj",
        ]
        for field in other_fields:
            sample[field] = self._read_hdf5_field(grp, field)

        # Override static_adj if global one is preferred by this helper (though unlikely for raw reader)
        if self.global_static_adj is not None and sample.get("static_adj") is None:
            sample["static_adj"] = self.global_static_adj

        return sample

    def get_session_id(self, idx):
        if not (0 <= idx < len(self.chunk_ids)):
            raise IndexError("Index out of bounds.")
        return self.chunk_ids[idx]

    def close(self):
        if self.h5file is not None:
            try:
                self.h5file.close()
                self.h5file = None
            except Exception as e:
                print(f"Error closing raw HDF5 reader: {e}")

    def __del__(self):
        self.close()


# --- ReadHDF5MultiModalData (Lazy loader for Preprocessing) ---
class ReadHDF5MultiModalData:  # This class is for PREPROCESSING

    def __init__(
        self, settings, original_h5_parent_path
    ):  # Path to dir containing train_converted.h5 etc.
        self.settings = settings
        self.history_steps = settings.history_steps
        self.num_nodes = settings.num_nodes
        self.dataset_name_prop = settings.dataset
        if not self.dataset_name_prop.endswith("Dataset"):
            self.dataset_name_prop += " Dataset"

        self.original_h5_base_path = original_h5_parent_path
        self.train_h5_file_path = os.path.join(self.original_h5_base_path, "train.h5")
        self.val_h5_file_path = os.path.join(self.original_h5_base_path, "val.h5")
        self.test_h5_file_path = os.path.join(self.original_h5_base_path, "test.h5")

        # Caches for on-demand loading, initialized to None
        self._raw_chunks_cache = {"train": None, "val": None, "test": None}
        self._logs_list_cache = {"train": None, "val": None, "test": None}
        self._idx2logs_indices_cache = {"train": None, "val": None, "test": None}
        # Add val/test log collection if FeatureExtractor.fit needs them (unlikely for fit)
        # or if TF-IDF is calculated over all splits together.
        print(f"ReadHDF5MultiModalData (for preprocessing) initialized. Paths:")
        print(f"  Train: {self.train_h5_file_path}")
        print(f"  Val:   {self.val_h5_file_path}")
        print(f"  Test:  {self.test_h5_file_path}")

    def _get_raw_chunk_reader_dataset(self, split_name):
        path = None
        if split_name == "train":
            path = self.train_h5_file_path
        elif split_name == "val":
            path = self.val_h5_file_path
        elif split_name == "test":
            path = self.test_h5_file_path

        if path and os.path.exists(path):
            return HDF5RawChunkReaderDataset(path, parse_log_keys_to_int=True)
        print(f"Warning: Raw HDF5 file for '{split_name}' not found at {path}")
        return None

    def _materialize_raw_chunks_from_hdf5(self, split_name):
        """
        Loads all raw chunks for a specified split into an in-memory dictionary.
        This is called by the respective property getter if the cache is empty.
        """
        # This method will fill the cache, e.g., self._raw_chunks_cache['train']
        cache_attr_name = (
            f"_raw_chunks_cache"  # The cache is a dict {'train': ..., 'val': ...}
        )

        if (
            getattr(self, cache_attr_name)[split_name] is None
        ):  # Check specific split cache
            print(f"Materializing ALL RAW chunks for '{split_name}' split from HDF5...")
            raw_chunk_reader = self._get_raw_chunk_reader_dataset(split_name)
            if not raw_chunk_reader:
                getattr(self, cache_attr_name)[
                    split_name
                ] = {}  # Cache empty dict if no data
                return getattr(self, cache_attr_name)[split_name]

            chunks_dict = {}
            for i in range(len(raw_chunk_reader)):
                # get_session_id() returns the HDF5 group name (original string chunk ID)
                chunk_id_key = raw_chunk_reader.get_session_id(i)
                # raw_chunk_reader[i] calls HDF5RawChunkReaderDataset.__getitem__
                chunks_dict[chunk_id_key] = raw_chunk_reader[i]
                if i > 0 and (i + 1) % 500 == 0:
                    print(
                        f"    Materialized {i+1}/{len(raw_chunk_reader)} raw chunks for '{split_name}'..."
                    )

            raw_chunk_reader.close()  # Close the temporary HDF5RawChunkReaderDataset file handle
            getattr(self, cache_attr_name)[split_name] = chunks_dict
            print(
                f"  Finished materializing {len(chunks_dict)} raw chunks for '{split_name}'."
            )

        return getattr(self, cache_attr_name)[split_name]

    def _build_log_indices_for_split(self, split_name):
        """
        Builds the flat list of all logs and the chunk_id->t->n->global_indices mapping.
        Called by properties if cache is empty.
        """
        logs_cache_attr = f"_logs_list_cache"
        indices_cache_attr = f"_idx2logs_indices_cache"

        if (
            getattr(self, logs_cache_attr)[split_name] is None
            or getattr(self, indices_cache_attr)[split_name] is None
        ):

            print(f"Building log list and indices for '{split_name}' split...")
            # Accessing the property (e.g., self.train_chunks) will trigger materialization if needed
            source_chunks_dict = getattr(self, f"{split_name}_chunks")

            if not source_chunks_dict:
                print(
                    f"  Cannot build log indices for '{split_name}': raw chunks data not available or empty."
                )
                getattr(self, logs_cache_attr)[split_name] = []
                getattr(self, indices_cache_attr)[split_name] = {}
                return

            all_logs_list = []
            idx2indices_dict = (
                {}
            )  # chunk_id_str -> timestep_key_parsed -> node_idx_parsed -> list of global indices
            current_global_log_start_idx = 0

            # chunk_id_key is string from HDF5 group name
            for chunk_id_key_str, chunk_data_content in source_chunks_dict.items():
                if chunk_id_key_str not in idx2indices_dict:
                    idx2indices_dict[chunk_id_key_str] = {}

                # chunk_data_content['logs'] is the raw log dict {t_parsed:{n_parsed:[str_list]}}
                logs_in_chunk = chunk_data_content.get("logs", {})

                for timestep_key_parsed in range(
                    self.history_steps
                ):  # Iterate 0 to history_steps-1
                    cur_ts_nodes_logs = logs_in_chunk.get(
                        timestep_key_parsed, {}
                    )  # Keys in logs_in_chunk are already parsed

                    if timestep_key_parsed not in idx2indices_dict[chunk_id_key_str]:
                        idx2indices_dict[chunk_id_key_str][timestep_key_parsed] = {}

                    for node_idx_parsed in range(
                        self.num_nodes
                    ):  # Iterate 0 to num_nodes-1
                        log_list_text = cur_ts_nodes_logs.get(
                            node_idx_parsed, []
                        )  # Keys are parsed

                        if (
                            node_idx_parsed
                            not in idx2indices_dict[chunk_id_key_str][
                                timestep_key_parsed
                            ]
                        ):
                            idx2indices_dict[chunk_id_key_str][timestep_key_parsed][
                                node_idx_parsed
                            ] = []

                        if log_list_text:  # If the list of log strings is not empty
                            all_logs_list.extend(log_list_text)
                            new_indices = list(
                                range(
                                    current_global_log_start_idx,
                                    current_global_log_start_idx + len(log_list_text),
                                )
                            )
                            idx2indices_dict[chunk_id_key_str][timestep_key_parsed][
                                node_idx_parsed
                            ].extend(new_indices)
                            current_global_log_start_idx += len(log_list_text)

            getattr(self, logs_cache_attr)[split_name] = all_logs_list
            getattr(self, indices_cache_attr)[split_name] = idx2indices_dict
            print(
                f"  Finished building log indices for '{split_name}'. Total logs collected: {len(all_logs_list)}."
            )

    def get_split_chunks(self, split_name):
        return self._materialize_raw_chunks_from_hdf5(split_name)

    @property
    def train_chunks(self):
        return self._materialize_raw_chunks_from_hdf5("train")

    @property
    def val_chunks(self):
        return self._materialize_raw_chunks_from_hdf5("val")

    @property
    def test_chunks(self):
        return self._materialize_raw_chunks_from_hdf5("test")

    @property
    def train_logs(self):
        if self._logs_list_cache["train"] is None:
            self._build_log_indices_for_split("train")
        return self._logs_list_cache["train"]

    @property
    def idx2logs_indices_train(self):
        if self._idx2logs_indices_cache["train"] is None:
            # Accessing train_logs property will trigger building both log list and indices
            _ = self.train_logs
        return self._idx2logs_indices_cache["train"]

    @property
    def val_logs(self):
        if self._logs_list_cache["val"] is None:
            self._build_log_indices_for_split("val")
        return self._logs_list_cache["val"]

    @property
    def idx2logs_indices_val(self):
        if self._idx2logs_indices_cache["val"] is None:
            _ = self.val_logs
        return self._idx2logs_indices_cache["val"]

    @property
    def test_logs(self):
        if self._logs_list_cache["test"] is None:
            self._build_log_indices_for_split("test")
        return self._logs_list_cache["test"]

    @property
    def idx2logs_indices_test(self):
        if self._idx2logs_indices_cache["test"] is None:
            _ = self.test_logs
        return self._idx2logs_indices_cache["test"]

    @property
    def dataset_name(self):
        return self.dataset_name_prop


# --- HDF5ProcessedDataset (PyTorch Dataset for PROCESSED HDF5 data) ---
class HDF5MultiModalDataset(Dataset):

    def __init__(self, processed_h5_file_path, settings):
        if not os.path.exists(processed_h5_file_path):
            raise FileNotFoundError(
                f"Processed HDF5 file not found: {processed_h5_file_path}"
            )
        self.processed_h5_file_path = processed_h5_file_path
        self.h5file = None
        self.settings = settings
        self.chunk_ids = []
        self._initialize_metadata()

    def _initialize_metadata(self):
        try:
            with h5py.File(self.processed_h5_file_path, "r") as temp_h5file:
                global_keys = []
                for key_in_h5 in temp_h5file.keys():
                    if key_in_h5 not in global_keys and isinstance(
                        temp_h5file[key_in_h5], h5py.Group
                    ):
                        self.chunk_ids.append(key_in_h5)
                if not self.chunk_ids:
                    raise ValueError(
                        f"No chunk data found in processed HDF5: {self.processed_h5_file_path}"
                    )
                try:
                    self.chunk_ids.sort(key=lambda x: int(x) if x.isdigit() else str(x))
                except ValueError:
                    self.chunk_ids.sort()

        except Exception as e:
            raise IOError(
                f"Metadata init error for processed HDF5 {self.processed_h5_file_path}: {e}"
            )

    def _ensure_file_open(self):
        if self.h5file is None:
            try:
                self.h5file = h5py.File(self.processed_h5_file_path, "r")
            except Exception as e:
                raise IOError(
                    f"Cannot open processed HDF5 {self.processed_h5_file_path}: {e}"
                )

    def __len__(self):
        return len(self.chunk_ids)

    def _read_field_from_processed_h5(self, grp, field_name, default_val=None):
        if field_name in grp:
            dataset = grp[field_name]
            if dataset.shape is None or dataset.shape == ():
                return dataset[()]
            else:
                return dataset[:]
        return (
            default_val
            if default_val is not None
            else (np.array([]) if field_name == "logs" else None)
        )

    def __getitem__(self, idx):
        self._ensure_file_open()
        if not (0 <= idx < len(self.chunk_ids)):
            raise IndexError("Index out of bounds.")
        chunk_id_str = self.chunk_ids[idx]
        try:
            grp = self.h5file[chunk_id_str]
        except KeyError:
            raise KeyError(f"Chunk ID '{chunk_id_str}' not found in processed HDF5.")

        sample = {"idx": idx}

        log_emb_data = self._read_field_from_processed_h5(grp, "log_emb")
        log_stats_data = self._read_field_from_processed_h5(grp, "log_stats")
        final_logs_feature = None

        if not self.settings.if_semantics:
            if log_stats_data is not None:
                final_logs_feature = log_stats_data
            else:
                # Fallback if log_stats is somehow missing even in non-semantic mode
                print(
                    f"Warning: if_semantics=False, but 'log_stats' missing for chunk {chunk_id_str}. 'logs' will be empty array."
                )
                final_logs_feature = np.array([])
        else:  # if_semantics is True
            if log_emb_data is None:
                # This implies an issue in preprocessing if semantics were on
                print(
                    f"Warning: if_semantics=True, but 'log_emb' missing for chunk {chunk_id_str}. Using zeros for log_emb part."
                )
                log_emb_data = np.zeros(
                    (
                        self.settings.history_steps,
                        self.settings.num_nodes,
                        self.settings.log_dim,
                    ),
                    dtype=np.float32,
                )

            if self.settings.if_log_feature:  # Concatenate log_emb and log_stats
                if (
                    log_stats_data is not None
                    and log_emb_data.size > 0
                    and log_stats_data.size > 0
                ):
                    # Keep log-count features consistent with masked log embeddings.
                    is_masked = np.all(log_emb_data == 0, axis=-1)

                    if np.any(is_masked):
                        # Broadcast the mask from (T, N) to log_stats shape.
                        mask_expanded = is_masked[..., np.newaxis]

                        log_stats_data = log_stats_data * (
                            1 - mask_expanded.astype(log_stats_data.dtype)
                        )
                    try:
                        final_logs_feature = np.concatenate(
                            (log_emb_data, log_stats_data), axis=-1
                        )
                    except ValueError as e:
                        print(
                            f"Concat error chunk {chunk_id_str}: emb_shape={log_emb_data.shape}, stats_shape={log_stats_data.shape}. Error: {e}. Using log_emb only."
                        )
                        final_logs_feature = log_emb_data
                elif log_emb_data.size > 0:
                    final_logs_feature = log_emb_data
                    if log_stats_data is None:
                        print(
                            f"Warning: if_log_feature=True, but 'log_stats' missing for chunk {chunk_id_str} to concat. Using only log_emb."
                        )
                elif log_stats_data is not None:
                    final_logs_feature = log_stats_data
                    print(
                        f"Warning: if_log_feature=True, but 'log_emb' missing/empty for chunk {chunk_id_str}. Using only log_stats for 'logs'."
                    )
                else:  # Both missing or log_emb was empty
                    final_logs_feature = (
                        np.array([]) if log_emb_data.size == 0 else log_emb_data
                    )
            else:  # if_log_feature is False, use only log_emb
                final_logs_feature = log_emb_data

        sample["logs"] = final_logs_feature

        other_fields = ["metrics", "traces", "adj", "y_labels", "labels"]
        # # If if_semantics is True, log_stats might be needed as a separate field by the model
        #     sample['log_stats'] = log_stats_data # It was loaded anyway

        for field in other_fields:
            sample[field] = self._read_field_from_processed_h5(grp, field)

        # Static Adjacency
        if "static_adj" in grp:
            sample["static_adj"] = self._read_field_from_processed_h5(grp, "static_adj")
        else:
            sample["static_adj"] = None

        return sample

    def get_session_id(self, idx):
        if not (0 <= idx < len(self.chunk_ids)):
            raise IndexError("Index out of bounds.")
        return self.chunk_ids[idx]

    def close(self):
        if self.h5file is not None:
            try:
                self.h5file.close()
                self.h5file = None
            except Exception as e:
                print(f"Error closing processed HDF5: {e}")

    def __del__(self):
        self.close()


# --- Preprocessing Function ---
def run_preprocessing(
    settings, device, data_path, original_dataset_root_path, processed_data_root_overall
):
    print(f"  Starting data preprocessing for dataset: {settings.dataset}")
    print(f"  Original dataset root: {original_dataset_root_path}")
    print(f"  Processed data will be saved under: {processed_data_root_overall}")

    # Ensure processed data root directory for this specific dataset exists
    # e.g. /path/to/data/TT-Eadro Dataset/processed/
    specific_processed_dataset_dir = processed_data_root_overall
    os.makedirs(specific_processed_dataset_dir, exist_ok=True)

    # Initialize reader for ORIGINAL HDF5 data
    # original_h5_parent_path is where train_converted.h5 etc. are
    # This should be like /path/to/data/TT-Eadro/processed
    original_h5_parent_path = original_dataset_root_path
    raw_data_reader = ReadHDF5MultiModalData(settings, original_h5_parent_path)

    dataset_name = settings.dataset
    if dataset_name.endswith("Dataset"):
        pass
    else:
        dataset_name += " Dataset"

    # Initialize FeatureExtractor
    word2vec_save_dir = data_path.replace("data", "word2vec", 1)
    specific_word2vec_save_dir = os.path.join(word2vec_save_dir, dataset_name)
    os.makedirs(word2vec_save_dir, exist_ok=True)

    dp_config = {
        "var_nums": None,
        "if_scale": True,
        "if_unlabel": True,
        "feature_type": "word2vec",
        "data_type": None,
        "log_window_size": None,
        "word2vec_model_type": "fasttext",  # fasttext_tfidf/bert
        "bert_model_name": "bert-base-uncased",
        "bert_batch_size": 64,
        "bert_max_length": 64,
        "pca_target_dim": settings.log_dim,
        "profile_log_encoding": True,
        "device": str(device),
        "word_embedding_dim": settings.log_dim,
        "word2vec_epoch": 50,
        "word2vec_save_dir": specific_word2vec_save_dir,
        "word_window": 3,
        "if_tfidf": settings.if_tfidf,
    }
    extractor = FeatureExtractor(settings, **dp_config)

    if dp_config["word2vec_model_type"] == "fasttext":
        need_fit_fasttext = not extractor.fasttext_model_exists()
        need_fit_tfidf = dp_config["if_tfidf"] and not extractor.tfidf_models_exist()

        if need_fit_fasttext or need_fit_tfidf:
            print("Fitting FeatureExtractor for FastText/TF-IDF...")
            extractor.fit(raw_data_reader)
        else:
            print("FastText/TF-IDF models already exist. Skipping fit().")
    elif dp_config["word2vec_model_type"] == "bert":
        print("BERT encoder selected. Skipping FeatureExtractor.fit().")
    else:
        raise ValueError(
            f"Unsupported log_encoder_type: {dp_config['word2vec_model_type']}"
        )

    if settings.if_perturb:
        perturber = PerturbationManager(settings, device)

    # Process and save each split to new HDF5 files
    for split in ["train", "val", "test"]:
        # Save to new HDF5 file
        processed_h5_file_path = os.path.join(
            specific_processed_dataset_dir, f"{split}_processed.h5"
        )
        if os.path.exists(processed_h5_file_path) and (
            split == "train" or split == "val"
        ):
            print(f"'{split}' processed h5 file exists, skipping the preprocess...")
            continue

        print(f"\nPreprocessing split: {split} for dataset {settings.dataset}")
        raw_chunks_dict_for_split = raw_data_reader.get_split_chunks(split)
        if not raw_chunks_dict_for_split:
            print(f"  No raw data for {split} split. Skipping.")
            continue
        # Always call transform to get log_emb (now that it's always calculated)
        print(f"  Transforming {split} data to get log_emb...")
        # FeatureExtractor.transform now returns a dict with 'log_emb', 'log_stats', and other fields.
        transformed_data_for_split = extractor.transform(
            raw_chunks_dict_for_split, datatype=split
        )

        print(f"  Saving processed {split} data to: {processed_h5_file_path}")
        with h5py.File(processed_h5_file_path, "w") as hf_proc:
            for (
                chunk_id_str,
                chunk_data_to_save,
            ) in transformed_data_for_split.items():  # Use transformed_data_for_split
                chunk_grp = hf_proc.create_group(
                    str(chunk_id_str)
                )  # Chunk ID is HDF5 group name

                if split == "test" and settings.if_perturb:
                    print(
                        f"  Processing perturbations for TEST split after FeatureExtractor..."
                    )
                    chunk_data_to_save = perturber.apply_perturbations_to_chunk(
                        chunk_data_to_save
                    )

                for (
                    field_key,
                    field_value,
                ) in chunk_data_to_save.items():  # field_value now includes log_emb
                    try:
                        data_to_write = field_value
                        if torch.is_tensor(data_to_write):
                            data_to_write = data_to_write.cpu().numpy()
                        elif isinstance(data_to_write, (list, tuple)):
                            # Check if it's a list of strings (raw logs) or a list/tuple of numbers
                            if (
                                field_key == "logs"
                            ):  # Original logs (text dict) should ideally not be here if transformed
                                print(
                                    f"Warning: Unexpected '{field_key}' type (list/tuple) for saving. Skipping for chunk {chunk_id_str}."
                                )
                                continue  # Skip if original logs are still list of dicts/lists of strings
                            else:  # Try to convert other lists/tuples to numpy arrays
                                try:
                                    data_to_write = np.asarray(data_to_write)
                                except Exception:
                                    pass  # Keep as list/tuple if cannot convert

                        if data_to_write is not None:
                            chunk_grp.create_dataset(field_key, data=data_to_write)
                    except TypeError as e_type:
                        print(
                            f"    TypeError saving field '{field_key}' for chunk {chunk_id_str} (type {type(field_value)}): {e_type}. Skipping field."
                        )
                    except Exception as e_save:
                        print(
                            f"    Error saving field '{field_key}' for chunk {chunk_id_str} in {split}: {e_save}"
                        )
    print("===== Data Preprocessing Complete =====")


def prepare_and_load_dataset(
    settings, device, dataset_name, data_path, sample_rate, if_scale=True, if_test=False
):
    """Prepare and load the released Eadro datasets."""
    if dataset_name.endswith("Dataset"):
        dataset_name = dataset_name
    else:
        dataset_name = dataset_name + " Dataset"

    if dataset_name not in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        raise ValueError(
            "This release keeps only the SN-Eadro and TT-Eadro dataset workflows."
        )

    dataset_path = os.path.join(data_path, dataset_name)
    word2vec_save_dir = data_path.replace("data", "word2vec")
    if not os.path.exists(word2vec_save_dir):
        os.makedirs(word2vec_save_dir)
    dataset_word2vec_save_dir = os.path.join(word2vec_save_dir, dataset_name)

    dataset = {}
    processed_dataset_path = os.path.join(dataset_path, "processed")
    print(f"Using preprocessed HDF5 workflow for {dataset_name}")

    train_processed_h5_path = os.path.join(processed_dataset_path, "train_processed.h5")
    val_processed_h5_path = os.path.join(processed_dataset_path, "val_processed.h5")
    test_processed_h5_path = os.path.join(processed_dataset_path, "test_processed.h5")

    processed_files_exist = (
        os.path.exists(train_processed_h5_path)
        and os.path.exists(val_processed_h5_path)
        and os.path.exists(test_processed_h5_path)
    )

    dp_config = {
        "var_nums": None,
        "if_scale": True,
        "if_unlabel": True,
        "feature_type": "word2vec",
        "data_type": None,
        "log_window_size": None,
        "word2vec_model_type": "fasttext",
        "word_embedding_dim": settings.log_dim,
        "word2vec_epoch": 50,
        "word2vec_save_dir": dataset_word2vec_save_dir,
        "word_window": 3,
        "if_tfidf": settings.if_tfidf,
    }

    temp_extractor = FeatureExtractor(settings, **dp_config)
    semantic_models_exist = temp_extractor.fasttext_model_exists() and (
        not settings.if_tfidf or temp_extractor.tfidf_models_exist()
    )

    run_preprocess_now = (
        settings.if_perturb or not processed_files_exist or not semantic_models_exist
    )

    if run_preprocess_now:
        print(f"Preprocessing required for {dataset_name}. Running now...")
        run_preprocessing(
            settings,
            device,
            data_path,
            processed_dataset_path,
            processed_dataset_path,
        )
    else:
        print(f"Found preprocessed HDF5 files and semantic models for {dataset_name}.")

    print("Creating PyTorch HDF5MultiModalDataset objects...")
    if os.path.exists(train_processed_h5_path):
        dataset["train"] = HDF5MultiModalDataset(train_processed_h5_path, settings)
    else:
        dataset["train"] = None
        print(f"Warning: {train_processed_h5_path} not found.")

    if os.path.exists(val_processed_h5_path):
        dataset["val"] = HDF5MultiModalDataset(val_processed_h5_path, settings)
    else:
        dataset["val"] = None
        print(f"Warning: {val_processed_h5_path} not found.")

    if os.path.exists(test_processed_h5_path):
        dataset["test"] = HDF5MultiModalDataset(test_processed_h5_path, settings)
    else:
        dataset["test"] = None
        print(f"Warning: {test_processed_h5_path} not found.")

    print("Dataset has been preprocessed and loaded!")
    return dataset

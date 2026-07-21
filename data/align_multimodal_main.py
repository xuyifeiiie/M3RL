"""
Build aligned multi-modal samples for M3RL.

The raw Eadro telemetry used in the paper contains metrics, logs, traces, fault
metadata, and service dependency information stored at different granularities.
This script aligns those modalities into fixed sliding windows. Each generated
sample contains metric features, log features, trace features, dynamic
adjacency matrices, system-level anomaly labels, and node-level root-cause
labels in the same `(sample, time, service, feature)` layout expected by the
M3RL dataloaders.
"""

import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import argparse
import tqdm
import shutil
import pickle
import random
import string
import numpy as np
from collections import defaultdict
from utils.utils import get_subfiles_subfolders


from data_utils import read_json, Dataset_Info, json_pretty_dump
from process_log import deal_logs
from process_trace import deal_traces, deal_ground_germany
from process_metric import deal_metrics

chunkids = set()
src = string.ascii_letters + string.digits


def get_chunkid():
    while True:
        chunkid = random.sample(src, 8)
        random.shuffle(chunkid)
        chunkid = "".join(chunkid)
        if chunkid not in chunkids:
            chunkids.add(chunkid)
            return chunkid


def set_sliding_windows_and_labels(dataset_info, cat_type, subdir_name, params):
    dataset_path = dataset_info.metadata["dataset_path"]
    history_steps = params["history_steps"]
    predict_steps = params["predict_steps"]
    threshold = params["threshold"]

    history_intervals = []
    predict_intervals = []

    if dataset_info.dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        # Eadro records provide fault metadata per experiment folder.
        subdir_name_split = subdir_name.split(".", 1)
        record_file_name = (
            subdir_name_split[0] + "." + "fault" + "-" + subdir_name_split[1]
        )
        records = read_json(
            os.path.join(dataset_path, cat_type, record_file_name + "_revised.json")
        )

        if len(records["faults"]) == 0:
            faults = []
        else:
            faults = [
                (
                    f_record["s"],
                    f_record["e"],
                    dataset_info.service2id[f_record["service"]],
                )
                for f_record in records["faults"]
            ]
        start, end = int(records["start"]), int(records["end"])
        time_steps = end - start + 1
    elif dataset_info.dataset_name == "Germany Dataset":
        # Germany records use separate concurrent/sequential experiment folders.
        records = read_json(
            os.path.join(
                dataset_path,
                cat_type,
                "IMPORTANT_experiment_start_end_data_revised.json",
            )
        )
        start = int(records["trace_start"])
        end = int(records["trace_end"])
        time_steps = int(end - start) + 1

        # labels is processed in deal_trace

    else:
        start = 0
        end = 0
        time_steps = 0

    ########## Generate annotated intervals ##########
    for i in range(start, end - history_steps - predict_steps + 1):
        history_intervals.append((i, i + history_steps - 1))
        predict_intervals.append(
            (i + history_steps, i + history_steps + predict_steps - 1)
        )

    if dataset_info.dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        labels = [-1] * len(history_intervals)
        # intervals for time stamp index
        for chunk_idx, (s, e) in enumerate(history_intervals):
            if len(faults) == 0:
                continue
            else:
                for fs, fe, culprit in faults:
                    overlap = 0
                    if s >= fs and s <= fe:
                        overlap = fe - s + 1
                    elif e >= fs and e <= fe:
                        overlap = e - fs + 1
                    if overlap >= threshold:
                        labels[chunk_idx] = culprit
                    if overlap > 0:
                        break

    elif dataset_info.dataset_name == "Germany Dataset":
        with open(
            os.path.join(
                dataset_path,
                "processed",
                cat_type,
                "traces",
                subdir_name + "_ground_labels.pkl",
            ),
            "rb",
        ) as f:
            ground = pickle.load(f)  # [action time steps, num_nodes] 0, 1, np.array

        labels = [
            np.zeros(ground.shape[1], dtype=int) for _ in range(len(history_intervals))
        ]
        for chunk_idx, (s, e) in enumerate(history_intervals):
            sub_labels = ground[chunk_idx : chunk_idx + history_steps, :]
            record = np.max(sub_labels, axis=0).astype(int)
            if_abnormal = np.any(record)
            if if_abnormal:
                pass
            labels[chunk_idx] = record

    print(
        "# starts at {}/{} and ends at {}/{}".format(
            history_intervals[0][0], start, history_intervals[-1][-1], end
        )
    )
    return history_intervals, predict_intervals, labels


def get_chunks_aligning_for_modalities(info, cat, subdir, params):
    """
    Organizes different modalities data into chunks aligned by time steps.

    Args:
        info(class Dataset_Info) : dataset info
        cat (str): fault type of data to be processed, e.g., 'fault' or 'faultless'
        subdir (str): sub directory name of the category if exists
        params (dict): Parameters for dataset generation.

    Returns:
        dict: Dictionary containing chunks of data.
    """
    task = params["task"]
    chunks = defaultdict(dict)
    tmp_folder = os.getcwd()
    if tmp_folder.endswith("data"):
        datafolder = tmp_folder
    else:
        datafolder = os.path.join(tmp_folder, "data")
    processed_path = os.path.join(datafolder, info.dataset_name, "processed")

    x_intervals, y_intervals, labels = set_sliding_windows_and_labels(
        info, cat, subdir, params
    )

    # trace data related
    if os.path.exists(os.path.join(processed_path, cat, subdir + "-traces.pkl")):
        with open(
            os.path.join(processed_path, cat, subdir + "-traces.pkl"), "rb"
        ) as fr:
            traces = pickle.load(fr)
            print("Loading traces successfully!")
    else:
        #     traces, labels = deal_traces(x_intervals, y_intervals, info, cat, subdir, params=params)
        traces = deal_traces(x_intervals, y_intervals, info, cat, subdir, params=params)
    # metric data related
    if os.path.exists(os.path.join(processed_path, cat, subdir + "-metrics.pkl")):
        with open(
            os.path.join(processed_path, cat, subdir + "-metrics.pkl"), "rb"
        ) as fr:
            metrics = pickle.load(fr)
            print("Loading metrics successfully!")
    else:
        metrics = deal_metrics(
            x_intervals, y_intervals, info, cat, subdir, params=params
        )
    # log data related
    if os.path.exists(os.path.join(processed_path, cat, subdir + "-logs.pkl")):
        with open(os.path.join(processed_path, cat, subdir + "-logs.pkl"), "rb") as fr:
            logs = pickle.load(fr)
            print("Loading logs successfully!")
    else:
        logs = deal_logs(x_intervals, y_intervals, info, cat, subdir, params=params)

    if info.dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        # dataset_name
        #      |--processed
        #             |--fault/faultless
        #                    |--logs
        #                         |--SN...-logs.pkl
        #                         |--SN...-logs.pkl
        #                         ...
        #                    |--trace
        #                         |--SN...-traces.pkl
        #                         |--SN...-traces.pkl
        #                         ...
        #                    |--metric
        #                         |--SN...-metrics.pkl
        #                         |--SN...-metrics.pkl
        #                         ...
        pass
    elif info.dataset_name == "Germany Dataset":
        # dataset_name
        #      |--processed
        #             |--concurrent data/sequential_data
        #                    |--logs
        #                         |--logs.pkl
        #                    |--trace
        #                         |--traces.pkl
        #                    |--metric
        #                         |--metrics.pkl
        pass
    else:
        print("Warning: Unseen Dataset!")
        exit(222)

    for idx in range(len(x_intervals)):
        chunk_id = get_chunkid()
        chunks[chunk_id]["logs"] = logs["logs"][idx]  # logs seqences
        chunks[chunk_id]["log_stats"] = logs["stats"][idx]
        chunks[chunk_id]["metrics"] = metrics["history_metrics"][idx]
        chunks[chunk_id]["traces"] = traces["metrics"][idx]
        chunks[chunk_id]["adj"] = traces["adj_matrix"][idx]
        chunks[chunk_id]["static_adj"] = traces["static_adj_matrix"]
        chunks[chunk_id]["y_labels"] = metrics["predict_metrics"][idx]
        chunks[chunk_id]["labels"] = labels[idx]
    return chunks


def generate_data_chunks(dataset_name, params):
    """
    Generate chunks for the given dataset name and parameters.

    Args:
        dataset_name (str): Name of the dataset. e.g.'SN-Eadro Dataset'
        params (dict): Parameters for dataset generation.

    Returns:
        dict: Dictionary containing chunks of data.
    """
    chunks = {}
    predict_steps = params["predict_steps"]
    history_steps = params["history_steps"]
    tmp_folder = os.getcwd()
    if tmp_folder.endswith("data"):
        datafolder = tmp_folder
    else:
        datafolder = os.path.join(tmp_folder, "data")

    info = Dataset_Info(dataset_name, os.path.join(datafolder, dataset_name))

    processed_path = os.path.join(datafolder, dataset_name, "processed")
    # Generate per-experiment chunks before merging train/val/test splits.
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    if dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        # Eadro chunks are stored under processed/fault and processed/faultless.
        for cat in ["fault", "faultless"]:
            cat_load_path = os.path.join(datafolder, dataset_name, cat)
            cat_save_path = os.path.join(processed_path, cat)
            if not os.path.exists(cat_save_path):
                os.makedirs(cat_save_path)
            _, subdirs = get_subfiles_subfolders(cat_load_path)
            for subdir in subdirs:
                if os.path.basename(subdir).startswith("SN.") or os.path.basename(
                    subdir
                ).startswith("TT."):
                    # Generate chunks for each subdir from original data directory
                    tmp_chunks = get_chunks_aligning_for_modalities(
                        info, cat, os.path.basename(subdir), params
                    )
                    # save chunks in save path
                    chunk_save_path = os.path.join(
                        cat_save_path, os.path.basename(subdir) + "-chunks.pkl"
                    )
                    chunks.update(tmp_chunks)
                    with open(chunk_save_path, "wb") as f:
                        pickle.dump(tmp_chunks, f)
    elif dataset_name == "Germany Dataset":
        # Germany chunks are stored under processed/concurrent data or sequential_data.
        for cat in ["concurrent data", "sequential_data"]:
            if cat == "concurrent data":
                flag = "concurrent"
            else:
                flag = "sequential"
            cat_save_path = os.path.join(processed_path, cat)
            if not os.path.exists(cat_save_path):
                os.makedirs(cat_save_path)
            tmp_chunks = get_chunks_aligning_for_modalities(info, cat, flag, params)
            chunk_save_path = os.path.join(cat_save_path, flag + "-chunks.pkl")
            chunks.update(tmp_chunks)
            with open(chunk_save_path, "wb") as f:
                pickle.dump(tmp_chunks, f)

    ############## Update info for metadata ##############
    info.insert_info("history_steps", history_steps)
    info.insert_info("predict_steps", predict_steps)
    info.insert_info("chunk_num", len(chunks))
    info.insert_info("edges", info.edges)
    sample_key = ""
    for key in chunks.keys():
        if key == "all_logs":
            continue
        else:
            sample_key = key
            break
    one_sample = chunks[sample_key]["log_stats"]
    if one_sample is not None:
        info.insert_info("event_num", one_sample.shape[-1])
    else:
        info.insert_info("event_num", 0)

    if os.path.exists(os.path.join(processed_path, "metadata.json")):
        os.remove(os.path.join(processed_path, "metadata.json"))
    json_pretty_dump(info.metadata, os.path.join(processed_path, "metadata.json"))

    return chunks


def generate_dataset(dataset_name, params):
    """
    Read data chunks files if they exist. If not, generate data chunks and split them into
    train val test dataset for the given dataset name and parameters.

    Args:
        dataset_name (str): Name of the dataset. e.g.'SN-Eadro Dataset'
        params (dict): Parameters for dataset generation.

    Returns:
        None
    """
    history_steps = params["history_steps"]
    predict_steps = params["predict_steps"]
    if_concat = params["concat"]
    if params["data_split_type"] == "normal":
        train_ratio = 0.8
        test_ratio = 0.1
        val_ratio = 0.1
    elif params["data_split_type"] == "minor_test":
        train_ratio = 0.9
        test_ratio = 0.01
        val_ratio = 1 - train_ratio - test_ratio
    else:
        train_ratio = 0.8
        test_ratio = 0.1
        val_ratio = 0.1
    print(
        f"***Beginning generating {dataset_name} for horizon steps {predict_steps} based on history steps {history_steps}...\n"
    )

    chunk_num = 0
    chunks = {}
    tmp_folder = os.getcwd()
    if tmp_folder.endswith("data"):
        datafolder = tmp_folder
    else:
        datafolder = os.path.join(tmp_folder, "data")
    processed_path = os.path.join(datafolder, dataset_name, "processed")
    train_data_path = os.path.join(processed_path, str(predict_steps) + "_train.npz")

    if not os.path.exists(processed_path) or not os.path.exists(train_data_path):
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        # Reuse processed chunks when available; otherwise regenerate them.
        chunks = generate_data_chunks(dataset_name, params)
    else:
        # Read chunks in ./{dataset_name}/processed/{category}/...-chunks.pkl.
        if dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
            # Eadro chunks are split by fault/faultless categories.
            for cat1 in ["fault", "faultless"]:
                cat_path = os.path.join(processed_path, cat1)
                if not os.path.exists(cat_path):
                    os.makedirs(cat_path)
                subfiles, _ = get_subfiles_subfolders(cat_path)
                for subfile in subfiles:
                    if (
                        os.path.basename(subfile).startswith("SN.")
                        or os.path.basename(subfile).startswith("TT.")
                    ) and (os.path.basename(subfile).endswith("-chunks.pkl")):
                        with open(subfile, "rb") as f:
                            chunks.update(pickle.load(f))
        elif dataset_name == "Germany Dataset":
            # Germany chunks are split by experiment type.
            for cat1 in ["concurrent data"]:
                cat_path = os.path.join(processed_path, cat1)
                subfiles, _ = get_subfiles_subfolders(cat_path)
                for subfile in subfiles:
                    if (
                        os.path.basename(subfile).startswith("concurrent")
                        or os.path.basename(subfile).startswith("sequential")
                    ) and (os.path.basename(subfile).endswith("-chunks.pkl")):
                        with open(subfile, "rb") as f:
                            chunks.update(pickle.load(f))

    chunk_num = len(chunks)
    category_groups = defaultdict(list)
    loaded_chunk_hashids = set()
    for root, subdirs, files in os.walk(processed_path):
        subdirs.sort()
        for filename in sorted(files):
            if filename.endswith("-chunks.pkl"):
                with open(os.path.join(root, filename), "rb") as f:
                    tmp_chunks = pickle.load(f)
                group_hashids = [k for k in tmp_chunks.keys() if k in chunks]
                if len(group_hashids) > 0:
                    relative_root = os.path.relpath(root, processed_path)
                    category = relative_root.split(os.sep)[0]
                    group_order = (1, filename)
                    if dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
                        subdir_name = filename[: -len("-chunks.pkl")]
                        subdir_name_split = subdir_name.split(".", 1)
                        record_file_name = (
                            subdir_name_split[0]
                            + "."
                            + "fault"
                            + "-"
                            + subdir_name_split[1]
                            + "_revised.json"
                        )
                        record_file_path = os.path.join(
                            datafolder,
                            dataset_name,
                            category,
                            record_file_name,
                        )
                        if os.path.exists(record_file_path):
                            records = read_json(record_file_path)
                            group_order = (0, int(records["start"]), filename)
                    category_groups[category].append(
                        (group_order, filename, group_hashids)
                    )
                    loaded_chunk_hashids.update(group_hashids)

    if loaded_chunk_hashids != set(chunks.keys()):
        raise ValueError("Some chunks cannot be matched to their experiment files.")

    ############## Split fault and faultless chunks chronologically ##############
    purge_num = history_steps + predict_steps - 1
    train_hashids = []
    val_hashids = []
    test_hashids = []
    purged_chunk_num = 0
    for category, groups in category_groups.items():
        category_hashids = []
        group_ends = []
        for group_order, group_name, group_hashids in sorted(groups):
            category_hashids.extend(group_hashids)
            group_ends.append(len(category_hashids))

        category_num = len(category_hashids)
        if category_num < 3:
            raise ValueError(
                "There are not enough chunks in {} to create train, val and "
                "test datasets.".format(category)
            )

        split_candidates = []
        for train_purge_num in range(purge_num + 1):
            for val_purge_num in range(purge_num + 1):
                available_num = (
                    category_num - train_purge_num - val_purge_num
                )
                train_num = int(train_ratio * available_num)
                val_num = int(val_ratio * available_num)
                test_num = int(available_num - train_num - val_num)
                if min(train_num, val_num, test_num) == 0:
                    continue

                train_end = train_num
                if train_end in group_ends:
                    required_train_purge_num = 0
                else:
                    current_group_end = min(x for x in group_ends if x > train_end)
                    required_train_purge_num = min(
                        purge_num, current_group_end - train_end
                    )
                if train_purge_num < required_train_purge_num:
                    continue

                val_start = train_end + train_purge_num
                val_end = val_start + val_num
                if val_end in group_ends:
                    required_val_purge_num = 0
                else:
                    current_group_end = min(x for x in group_ends if x > val_end)
                    required_val_purge_num = min(
                        purge_num, current_group_end - val_end
                    )
                if val_purge_num < required_val_purge_num:
                    continue

                test_start = val_end + val_purge_num
                if test_start >= category_num:
                    continue
                split_candidates.append(
                    (
                        train_purge_num + val_purge_num,
                        train_end,
                        val_start,
                        val_end,
                        test_start,
                    )
                )

        if len(split_candidates) == 0:
            raise ValueError(
                "Cannot create non-overlapping train, val and test datasets "
                "for {}.".format(category)
            )

        split_candidates.sort(key=lambda x: x[0])
        (
            category_purged_chunk_num,
            train_end,
            val_start,
            val_end,
            test_start,
        ) = split_candidates[0]

        train_hashids.extend(category_hashids[:train_end])
        val_hashids.extend(category_hashids[val_start:val_end])
        test_hashids.extend(category_hashids[test_start:])
        purged_chunk_num += category_purged_chunk_num

    print(
        "# Remove {} boundary chunks to preserve split ratios and avoid "
        "overlap between datasets.".format(purged_chunk_num)
    )

    train_chunks = {k: chunks[k] for k in train_hashids}
    val_chunks = {k: chunks[k] for k in val_hashids}
    test_chunks = {k: chunks[k] for k in test_hashids}

    for cat2 in ["train", "val", "test"]:
        tmp_chunks = locals()[cat2 + "_chunks"]
        save_path = os.path.join(
            datafolder, dataset_name, "processed", "{}.npz".format(cat2)
        )
        np.savez(
            save_path,
            data=tmp_chunks,
        )


def check_dataset(dataset_name, params):
    """
    Check dataset if it exists. If not exists, generate the dataset.

    Args:
        dataset_name (str): Name of the dataset. e.g.'SN-Eadro Dataset'
        params (dict): Parameters for dataset generation.

    Returns:
        None
    """
    predict_steps = params["predict_steps"]
    print(f"***Checking {dataset_name}...")

    tmp_folder = os.getcwd()
    if tmp_folder.endswith("data"):
        datafolder = tmp_folder
    else:
        datafolder = os.path.join(tmp_folder, "data")
    processed_dataset_path = os.path.join(datafolder, dataset_name, "processed")
    if (
        os.path.exists(
            os.path.join(processed_dataset_path, str(predict_steps) + "_train.npz")
        )
        and os.path.exists(
            os.path.join(processed_dataset_path, str(predict_steps) + "_val.npz")
        )
        and os.path.exists(
            os.path.join(processed_dataset_path, str(predict_steps) + "_test.npz")
        )
    ):
        print(f"***{dataset_name} for horizon {predict_steps} has been processed.")
    else:
        print(f"***{dataset_name} for horizon {predict_steps} has not been found...\n")
        print(f"***Begining to generate npz files for {dataset_name}...\n")
        generate_dataset(dataset_name, params)
        print(
            f"***{dataset_name} for horizon {predict_steps} has been successfully generated.\n"
        )

    ########## Print statistics ##########
    metadata_file_path = os.path.join(processed_dataset_path, "metadata.json")
    metadata = read_json(metadata_file_path)
    print(f"***Metadata for {dataset_name}:\n")
    # [S, T, N, F], S is number of samples, T is number of time steps, N is number of instances, F is number of features
    print(
        "Number of samples for {} is : {}\n".format(
            dataset_name, str(metadata["chunk_num"])
        )
    )


parser = argparse.ArgumentParser()
parser.add_argument("--concat", action="store_true")
parser.add_argument("--delete_all", default=False, action="store_true")
parser.add_argument(
    "--delete",
    action="store_true",
    default=False,
    help="just remove the chosen dataset",
)
parser.add_argument(
    "--if_process_all",
    type=eval,
    default=False,
    help="Whether to process all raw datasets",
)
parser.add_argument("--threshold", default=1, type=int)
parser.add_argument("--history_steps", default=16, type=int)
parser.add_argument("--predict_steps", default=16, type=int)
parser.add_argument("--dataset_index", default=5, type=int, help="The dataset index")
parser.add_argument("--task", default="forecast", type=str)
parser.add_argument(
    "--data_split_type", type=str, default="normal", help="set data split ratio"
)
parser.add_argument(
    "--log_process_type",
    type=str,
    default="template_statistics",
    help="set log process type",
)
params = vars(parser.parse_args())

if "__main__" == __name__:
    """
    {0: 'Germany Dataset', 1: 'Hades Dataset', 2: 'MicroSS Dataset', 3: 'OB-Nezha Dataset', 4: 'SN-Eadro Dataset',
    5: 'TT-Eadro Dataset', 6: 'TT-Ebpf Dataset', 7: 'TT-Nezha Dataset'}
    """
    dataset_name_l = [
        "Germany",
        "Hades",
        "MicroSS",
        "OB-Nezha",
        "SN-Eadro",
        "TT-Eadro",
        "TT-Ebpf",
        "TT-Nezha",
    ]
    dataset_name_chosen = dataset_name_l[params["dataset_index"]] + " Dataset"
    idx_to_dataset_name = {idx: x for idx, x in enumerate(dataset_name_l)}
    print(idx_to_dataset_name)
    dataset_chosen_processed_path = os.path.join(
        os.getcwd(), dataset_name_chosen, "processed"
    )
    if params["delete_all"]:
        _input = input(
            "Do you really want to delete all previous files?! Input yes if you are so confident.\n"
        )
        flag = _input.lower() == "yes"
        for name in dataset_name_l:
            dataset_name_tmp = name + "Dataset"
            dataset_processed_path_tmp = os.path.join(
                os.getcwd(), dataset_name_tmp, "processed"
            )
            if (
                flag
                and os.path.exists(dataset_processed_path_tmp)
                and len(dataset_processed_path_tmp) > 2
            ):
                shutil.rmtree(dataset_processed_path_tmp)
            else:
                pass
        print("***Deleting completed...")
    if params["delete"] and os.path.exists(dataset_chosen_processed_path):
        shutil.rmtree(dataset_chosen_processed_path)

    if params["if_process_all"]:
        for k, v in tqdm.tqdm(idx_to_dataset_name.items()):
            dataset_name_tmp = v + " Dataset"
            print(f"Beginning to process {dataset_name_tmp}...")
            check_dataset(dataset_name=dataset_name_tmp, params=params)
        print("All datasets have been processed successfully!")
    else:
        print(f"Beginning to process {dataset_name_chosen}...")
        check_dataset(dataset_name=dataset_name_chosen, params=params)

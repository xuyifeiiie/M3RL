"""
Run the first preprocessing pass for raw telemetry modalities.

This entry point prepares per-modality processed files. It does not create the
final M3RL training windows directly; instead it calls the log, metric, and
trace processors so `align_multimodal_main.py` can later combine them into
synchronized metric/log/trace samples and labels.
"""

import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import json
import csv
import argparse
import tqdm


from utils.utils import get_subfiles_subfolders
from data_utils import Dataset_Info
from process_log import process_log
from process_trace import process_trace
from process_metric import process_metric


def preprocess_dataset(args, dataset_name, dataset_path):
    """Preprocess raw logs, metrics, and traces for one dataset folder."""
    dataset_info = Dataset_Info(dataset_name, dataset_path)
    print("Begining to preprocess log data for {} Dataset".format(dataset_name))
    process_log(dataset_info)
    print(
        "Log data has been preprocessed for {} Dataset successfully!".format(
            dataset_name
        )
    )
    print("Begining to preprocess metric data for {} Dataset".format(dataset_name))
    process_metric(dataset_info)
    print(
        "Metric data has been preprocessed for {} Dataset successfully!".format(
            dataset_name
        )
    )
    print("Begining to preprocess trace data for {} Dataset".format(dataset_name))
    process_trace(dataset_info)
    print(
        "Trace data has been preprocessed for {} Dataset successfully!".format(
            dataset_name
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./",
        help="Raw Data.",
    )
    parser.add_argument(
        "--dataset_index",
        type=int,
        default=2,
        help="Dataset index.",
    )
    parser.add_argument(
        "--if_process_all",
        type=eval,
        default=False,
        help="Whether to process all raw datasets",
    )
    args = parser.parse_args()
    params = vars(args)
    print(params)

    tmp_folder = os.getcwd()
    if tmp_folder.endswith("data"):
        datafolder = tmp_folder
    else:
        datafolder = os.path.join(tmp_folder, "data")

    _, dataset_list = get_subfiles_subfolders(datafolder)
    dataset_list = [item for item in dataset_list if "Dataset" in item]
    idx_to_dataset_name = {
        idx: os.path.basename(x) for idx, x in enumerate(dataset_list)
    }
    print(idx_to_dataset_name)

    if args.if_process_all:
        for k, v in tqdm.tqdm(idx_to_dataset_name.items()):
            print(f"Beginning to process {v}...")
            preprocess_dataset(args, v, dataset_list[k])
            print(f"{v} has been processed successfully!")
        print("All datasets have been processed successfully!")
    else:
        dataset_chosen = idx_to_dataset_name[args.dataset_index]
        print(f"Beginning to process {dataset_chosen}...")
        preprocess_dataset(args, dataset_chosen, dataset_list[args.dataset_index])
        print(f"{dataset_chosen} has been processed successfully!")

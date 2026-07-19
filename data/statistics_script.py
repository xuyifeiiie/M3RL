import os

"""Small dataset-inspection helpers for processed M3RL artifacts.

These utilities are not part of Stage 1 pre-training or Stage 2 downstream
training. They summarize processed arrays, labels, and graph statistics so the
released data pipeline can be sanity-checked before running experiments.
"""

import sys
import h5py
import pandas as pd
import argparse
from tqdm import tqdm


def get_subfiles_subfolders(path):
    """
    Helper function to get subfiles and subfolders.
    """
    subfiles, subfolders = [], []
    for f in os.scandir(path):
        if f.is_dir():
            subfolders.append(f.path)
        else:
            subfiles.append(f.path)
    return subfiles, subfolders


def analyze_dataset_labels(h5_file_path):
    """
    Analyzes a single HDF5 file to count normal and abnormal samples.

    Args:
        h5_file_path (str): The path to the HDF5 file.

    Returns:
        tuple: A tuple containing (normal_count, abnormal_count, total_count).
               Returns (0, 0, 0) if the file cannot be processed.
    """
    if not os.path.exists(h5_file_path):
        print(f"Warning: File not found at {h5_file_path}. Skipping.")
        return 0, 0, 0

    normal_count = 0
    abnormal_count = 0
    abnormal_idxs = []
    count = 0
    try:
        with h5py.File(h5_file_path, "r") as hf:
            # Filter for groups (chunks) only, excluding other datasets like 'static_adj_global'
            chunk_ids = [key for key in hf.keys() if isinstance(hf[key], h5py.Group)]

            # Use tqdm for a progress bar
            for chunk_id in tqdm(
                chunk_ids,
                desc=f"Analyzing {os.path.basename(h5_file_path)}",
                leave=False,
            ):
                try:
                    # Access the 'labels' dataset within the chunk group
                    label = hf[chunk_id]["labels"][()]

                    if label == -1:
                        normal_count += 1
                    elif label >= 0:
                        abnormal_count += 1
                        if "test" in h5_file_path:
                            abnormal_idxs.append(count)
                            count = count + 1
                except KeyError:
                    print(
                        f"Warning: 'labels' dataset not found in chunk '{chunk_id}' of file '{h5_file_path}'. Skipping chunk."
                    )
                except Exception as e:
                    print(f"An error occurred processing chunk '{chunk_id}': {e}")

    except Exception as e:
        print(f"Error opening or reading HDF5 file {h5_file_path}: {e}")
        return 0, 0, 0

    total_count = normal_count + abnormal_count
    return normal_count, abnormal_count, total_count, abnormal_idxs


def main():
    """
    Main function to run the statistics script.
    """
    parser = argparse.ArgumentParser(
        description="Count normal and abnormal samples in HDF5 datasets."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./",
        help="Path to the directory containing dataset folders.",
    )
    parser.add_argument(
        "--dataset_index",
        type=int,
        default=6,
        help="Index of the dataset folder to analyze.",
    )
    args = parser.parse_args()

    # --- Dynamically find dataset folders ---
    _, dataset_folders = get_subfiles_subfolders(args.data_path)
    dataset_folders = sorted(
        [item for item in dataset_folders if "Dataset" in os.path.basename(item)]
    )
    idx_to_dataset_info = {
        idx: {"name": os.path.basename(path), "path": path}
        for idx, path in enumerate(dataset_folders)
    }

    if not idx_to_dataset_info:
        print(f"Error: No dataset folders found in '{args.data_path}'.")
        sys.exit(1)

    print("Available datasets:")
    for idx, info in idx_to_dataset_info.items():
        print(f"  {idx}: {info['name']}")

    if args.dataset_index not in idx_to_dataset_info:
        print(
            f"Error: Invalid dataset_index '{args.dataset_index}'. Please choose from the list above."
        )
        sys.exit(1)

    selected_dataset = idx_to_dataset_info[args.dataset_index]
    print(f"\nSelected dataset: {selected_dataset['name']}\n")

    processed_data_path = os.path.join(selected_dataset["path"], "processed")

    if not os.path.isdir(processed_data_path):
        print(
            f"Error: 'processed' directory not found in '{selected_dataset['path']}'."
        )
        sys.exit(1)

    # --- Analysis ---
    stats = {}
    file_names = {
        "Train": "train_processed.h5",
        "Validation": "val_processed.h5",
        "Test": "test_processed.h5",
    }

    for name, filename in file_names.items():
        h5_path = os.path.join(processed_data_path, filename)
        normal, abnormal, total, abnormal_idxs = analyze_dataset_labels(h5_path)
        stats[name] = {
            "Normal (-1)": normal,
            "Abnormal (>=0)": abnormal,
            "Total": total,
            "Abnormal ids": abnormal_idxs,
        }

    # --- Display Results using Pandas DataFrame for nice formatting ---
    df = pd.DataFrame.from_dict(stats, orient="index")

    # Calculate totals
    df.loc["Total"] = df.sum()

    # Calculate percentages
    df["Normal %"] = (df["Normal (-1)"] / df["Total"] * 100).map("{:.2f}%".format)
    df["Abnormal %"] = (df["Abnormal (>=0)"] / df["Total"] * 100).map("{:.2f}%".format)

    # Reorder columns for better readability
    df = df[["Total", "Normal (-1)", "Abnormal (>=0)", "Normal %", "Abnormal %"]]

    print("\n--- Label Statistics ---")
    print(df.to_string())
    print("-" * 22)


if __name__ == "__main__":
    main()

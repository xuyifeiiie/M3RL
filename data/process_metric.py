"""
Preprocess raw metric telemetry for M3RL.

Metrics provide the numerical service state sequence used by both the Stage 1
prediction objective and the Stage 2 forecasting task. This module copies and
normalizes raw per-service metric CSV files into the processed dataset layout
that later alignment code can index by time window and service id.
"""

import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import pandas as pd
import numpy as np
import csv
import json
import pickle
import time
from datetime import datetime
from logparser import Drain
from utils.utils import get_subfiles_subfolders


def process_metric(dataset_info):
    """Prepare per-service metric CSV files for each experiment split."""
    dataset_name = dataset_info.dataset_name
    metric_data_path = dataset_info.metadata["data_path"]["metrics"]
    data_processed_path = os.path.join(dataset_info.dataset_path, "processed")
    if not os.path.exists(data_processed_path):
        os.makedirs(data_processed_path)

    if dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        for key, value in metric_data_path.items():
            # Eadro metric files are organized by fault/faultless experiment folders.
            if key == "fault":
                metric_processed_dir = os.path.join(
                    data_processed_path, "fault", "metrics"
                )
                if not os.path.exists(metric_processed_dir):
                    os.makedirs(metric_processed_dir)
            else:
                metric_processed_dir = os.path.join(
                    data_processed_path, "faultless", "metrics"
                )
                if not os.path.exists(metric_processed_dir):
                    os.makedirs(metric_processed_dir)
            for metric_path in value:
                if sys.platform.startswith("win"):
                    subcat = metric_path.split("\\")[-3]
                    service = os.path.basename(metric_path).split(".")[0]
                else:
                    subcat = metric_path.split("/")[-3]
                    service = os.path.basename(metric_path).split(".")[0]
                with open(metric_path, "r", newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = [row for row in reader]
                    df = pd.DataFrame(rows)
                    df.to_csv(
                        os.path.join(
                            metric_processed_dir, subcat + "-" + service + ".csv"
                        ),
                        index=False,
                    )

    elif dataset_name == "Germany Dataset":
        for key, value in metric_data_path.items():
            # Germany metric files are organized by concurrent/sequential folders.
            if key == "con":
                start = dataset_info.metadata[
                    "concurrent data_IMPORTANT_experiment_start_end_data"
                ]["trace_start"]
                end = dataset_info.metadata[
                    "concurrent data_IMPORTANT_experiment_start_end_data"
                ]["trace_end"]
                metric_processed_dir = os.path.join(
                    data_processed_path, "concurrent data", "metrics"
                )
                if not os.path.exists(metric_processed_dir):
                    os.makedirs(metric_processed_dir)
            else:
                start = dataset_info.metadata[
                    "sequential_data_IMPORTANT_experiment_start_end_data"
                ]["trace_start"]
                end = dataset_info.metadata[
                    "sequential_data_IMPORTANT_experiment_start_end_data"
                ]["trace_end"]
                metric_processed_dir = os.path.join(
                    data_processed_path, "sequential_data", "metrics"
                )
                if not os.path.exists(metric_processed_dir):
                    os.makedirs(metric_processed_dir)
            for metric_path in value:
                service = os.path.basename(metric_path).split("_")[0]
                df = pd.read_csv(metric_path)
                df.set_index("now", inplace=True)

                df.index = df.index.map(lambda x: x[:-5])
                df.index = df.index.map(
                    lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))
                    - 60 * 60
                )
                df = df.groupby("now").mean()

                # Detect missing timestamps before interpolation/alignment.
                full_index = pd.RangeIndex(
                    start=df.index.min(), stop=df.index.max() + 1, step=1
                )  # Complete timestamp range for the current service metric.
                missing_timestamps = full_index.difference(df.index)
                # Create placeholder rows for missing timestamps.
                missing_df = pd.DataFrame(index=missing_timestamps, columns=df.columns)
                df = pd.concat([df, missing_df]).sort_index()
                df = df.interpolate(method="linear")

                df = df[
                    ["cpu.user", "mem.used", "load.min1", "load.min5", "load.min15"]
                ]
                for name in df.columns.values:
                    df[name] = (df[name] - df[name].min()) / (
                        df[name].max() - df[name].min()
                    )
                df = df.fillna(0)
                df = df.dropna(axis=0, how="any")
                df.reset_index(drop=False)
                df.rename(columns={"index": "now"}, inplace=True)
                df = df.loc[(df.index >= start) & (df.index <= end), :]
                df.sort_index(axis=0)
                df.to_csv(
                    os.path.join(metric_processed_dir, service + "_metric.csv"),
                    index=True,
                )


z_zero_scaler = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8)


def deal_metrics(x_intervals, y_intervals, info, cat, subdir, params):
    print(
        "***No metrics.pkl file of {} founded, beginning to deal with preprocess metrics...".format(
            subdir
        )
    )
    tmp_folder = os.getcwd()
    if tmp_folder.endswith("data"):
        datafolder = tmp_folder
    else:
        datafolder = os.path.join(tmp_folder, "data")
    processed_path = os.path.join(datafolder, info.dataset_name, "processed")
    history_steps = params["history_steps"]
    predict_steps = params["predict_steps"]

    print("*** Dealing with metrics of {}...".format(subdir))
    metric_num = len(info.metric_names)
    metrics = {}
    services_list = sorted(info.service_names)

    if info.dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        history_metrics = np.zeros(
            (len(x_intervals), info.node_num, history_steps, metric_num),
            dtype=np.float32,
        )
        predict_metrics = np.zeros(
            (len(y_intervals), info.node_num, predict_steps, metric_num),
            dtype=np.float32,
        )
        for nid, service in enumerate(services_list):
            df = pd.read_csv(
                os.path.join(
                    processed_path, cat, "metrics", subdir + "-" + service + ".csv"
                )
            )
            df[info.metric_names] = df[info.metric_names].apply(z_zero_scaler)
            df.set_index(["timestamp"], inplace=True)
            for chunk_idx, (s, e) in enumerate(x_intervals):
                values = df.loc[s:e, :].to_numpy()
                assert values.shape == (
                    history_steps,
                    metric_num,
                ), "{} shape in {}--{}".format(values.shape, s, e)
                history_metrics[chunk_idx, nid, :, :] = values
            for chunk_idx, (s, e) in enumerate(y_intervals):
                values = df.loc[s:e, :].to_numpy()
                assert values.shape == (
                    predict_steps,
                    metric_num,
                ), "{} shape in {}--{}".format(values.shape, s, e)
                predict_metrics[chunk_idx, nid, :, :] = values

    elif info.dataset_name == "Germany Dataset":
        metric_num = metric_num - 1
        history_metrics = np.zeros(
            (len(x_intervals), info.node_num, history_steps, metric_num),
            dtype=np.float32,
        )
        predict_metrics = np.zeros(
            (len(y_intervals), info.node_num, predict_steps, metric_num),
            dtype=np.float32,
        )
        for nid, service in enumerate(services_list):
            df = pd.read_csv(
                os.path.join(processed_path, cat, "metrics", service + "_metric.csv")
            )
            df.rename(columns={"Unnamed: 0": "now"}, inplace=True)
            df.set_index(["now"], inplace=True)
            for chunk_idx, (s, e) in enumerate(x_intervals):
                values = df.loc[s:e, :].to_numpy()
                assert values.shape == (
                    history_steps,
                    metric_num,
                ), "{} shape in {}--{}".format(values.shape, s, e)
                history_metrics[chunk_idx, nid, :, :] = values
            for chunk_idx, (s, e) in enumerate(y_intervals):
                values = df.loc[s:e, :].to_numpy()
                assert values.shape == (
                    predict_steps,
                    metric_num,
                ), "{} shape in {}--{}".format(values.shape, s, e)
                predict_metrics[chunk_idx, nid, :, :] = values

    metrics["history_metrics"] = np.transpose(history_metrics, (0, 2, 1, 3))  # B T N F
    metrics["predict_metrics"] = np.transpose(predict_metrics, (0, 2, 1, 3))

    with open(os.path.join(processed_path, cat, subdir + "-metrics.pkl"), "wb") as fw:
        pickle.dump(metrics, fw)
    return metrics

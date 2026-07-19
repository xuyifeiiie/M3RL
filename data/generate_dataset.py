"""
Generic dataset generation utilities.

Most M3RL experiments use the Eadro-specific multi-modal HDF5 workflow, but
this file keeps the generic NPZ windowing utilities used by earlier datasets
and ablation-style inputs. The functions construct train/validation/test
windows, normalize numeric series, and save scaler metadata without changing
the model-side computation.
"""

import csv
import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import argparse
import numpy as np
import pandas as pd
import tqdm
from utils.utils import (
    scan_category,
    get_adjacency_matrix,
    get_subfiles_subfolders,
    standard_transform,
)


def compute_reshape_mean_std_for_4_dimension_data(data):
    """
    :param data: S*T*N*D   4 dimension dataset
    :return: step_mean: S*TN*D
             step_std: S*TN*D
    """
    a, b, c, d = data.shape
    # Flatten samples, time steps, and nodes while preserving feature dimension.
    reshape_data = data.reshape(-1, d)
    step_mean = reshape_data.mean(axis=0)
    step_std = reshape_data.std(axis=0)
    return step_mean, step_std


def normalize_dataset(raw_data, normalize):
    # Normalize with the selected legacy strategy.
    t, n, f = raw_data.shape
    raw_data = raw_data.reshape(-1, f)  # tn f
    _, f = raw_data.shape
    scale = np.ones(f)
    data = np.zeros(raw_data.shape)

    if normalize == 0:
        data = raw_data
        data = data.reshape(t, n, f)

    if normalize == 1:
        data = raw_data / np.max(raw_data, axis=0)
        data = data.reshape(t, n, f)

    if normalize == 2:
        scale = np.max(np.abs(raw_data), axis=0)
        data = raw_data / np.max(np.abs(raw_data), axis=0)
        data = data.reshape(t, n, f)

    return data, scale


def preprocess_df(df):
    """Clean low-level metric columns and convert selected network units."""
    df.drop(["NodeCpuStat_LogicalCores", "NodeNetStat_Up"], axis=1, inplace=True)
    for column in df.columns.tolist():
        if column.startswith("CNet"):
            # Convert nanosecond-level network metrics to millisecond scale.
            if column in [
                "CNetConnectionDuration",
                "CNetConnectionLatency",
                "CNetPassiveEstablishSpan",
                "CNetSingleRecvSpanNs",
                "CNetSingleSendSpanNs",
                "CNetSingleTransLatencyNs",
                "CNetStateSpanFromClose",
                "CNetStateSpanFromSynrecv",
            ]:
                df[column] = df[column] / 1000000
            if column == "CNetRttVar":
                df[column] = df[column] / 200000
            elif column in [
                "CNetConnectionDuration",
                "CNetSingleTransLatencyNs",
                "CNetPassiveEstablishSpan",
                "CNetRecvPackets",
                "CNetRecvBytes",
                "CNetSentBytes",
                "CNetSentPackets",
                "CNetStateSpanFromClose",
                "CNetStateSpanFromSynrecv",
            ]:
                df[column] = df[column].apply(np.log1p)
        if column in [
            "JvmStat_Size",
            "JvmStat_Used",
            "JvmStat_JvmGCTime",
            "JvmStat_JvmSafepointSyncTime",
            "JvmStat_JvmSafepointTime",
            "NodeDiskStat_BytesReadVar",
            "NodeDiskStat_BytesWrittenVar",
            "NodeMemStat_AvailableBytes",
            "NodeMemStat_CachedBytes",
            "NodeMemStat_FreeBytes",
            "NodeMemStat_TotalBytes",
            "NodeNetStat_RxBytesVar",
            "NodeNetStat_RxPacketsVar",
            "NodeNetStat_TxBytesVar",
            "NodeNetStat_TxPacketsVar",
        ]:
            df[column] = df[column].apply(np.log1p)
    return df


def update_min_max(old_min, old_max, new_min, new_max):
    assert len(old_min) == len(new_min)
    assert len(old_max) == len(new_max)
    # min
    for i in range(len(old_min)):
        #     continue
        if new_min[i] <= old_min[i]:
            old_min[i] = new_min[i]
        else:
            continue
    # max
    for i in range(len(old_max)):
        #     continue
        if new_max[i] >= old_max[i]:
            old_max[i] = new_max[i]
        else:
            continue
    return old_min, old_max


def generate_graph_seq2seq_io_data_for_trainticket(
    category_metric_df,
    category_instance_df,
    category_connection_df,
    x_offsets,
    y_offsets,
    adj_in_offsets,
    adj_out_offsets,
    sample_rate,
    num_of_instances,
):
    """
    Generate samples in the pattern of H history steps dataset mapping the P predict steps
    In this function, P = H. And P[0] is the next time stamp of H[-1].
    :param category_metric_df:
    :param category_instance_df:
    :param category_connection_df:
    :param x_offsets:
    :param y_offsets:
    :param adj_in_offsets:
    :param adj_out_offsets:
    :param sample_rate:
    :param num_of_instances:

    :return:
    # x: (time_steps, input_length, num_nodes, input_dim)
    # y: (time_steps, output_length, num_nodes, output_dim)

    """

    num_samples, num_features = category_metric_df.shape
    x, y = [], []
    l1, l2, l3 = [], [], []
    adj_in, adj_out = [], []
    adj_list = []

    data = category_metric_df.values

    tmp_data = data[:, 4:-3]
    arr_min = tmp_data.min(axis=0)
    arr_max = tmp_data.max(axis=0)

    time_steps = num_samples / num_of_instances
    data = np.delete(data, (0, 1, 2, 3), axis=1)
    data = data.reshape(int(time_steps), num_of_instances, -1)

    min_t = abs(min(x_offsets))  # -63--0  63
    max_t = abs(int(time_steps) - abs(max(y_offsets)))  # 1--64  130-64 = 66
    for t in range(min_t, max_t):
        x.append(data[(t + x_offsets), :, :-3])
        y.append(data[(t + y_offsets), :, :-3])
        l1.append(data[(t + y_offsets), :, -3:-2])
        l2.append(data[(t + y_offsets), :, -2:-1])
        l3.append(data[(t + y_offsets), :, -1:])

    start_stamp = category_metric_df.iloc[0]["TimeStamp"]
    end_stamp = category_metric_df.iloc[-1]["TimeStamp"]
    tmp_timestamp = start_stamp - int(sample_rate)
    while tmp_timestamp + int(sample_rate) <= end_stamp:
        span_df = category_connection_df[
            (category_connection_df["TimeStamp"] > tmp_timestamp)
            & (category_connection_df["TimeStamp"] <= tmp_timestamp + int(sample_rate))
        ]
        adj = get_adjacency_matrix("", span_df, num_of_instances, type_="counts")
        adj_list.append(adj)
        tmp_timestamp = tmp_timestamp + int(sample_rate)

    adj_array = np.array(adj_list)

    for t in range(min_t, max_t):
        adj_in.append(adj_array[(t + adj_in_offsets)])
        adj_out.append(adj_array[(t + adj_out_offsets)])

    x_array = np.stack(x, axis=0)
    y_array = np.stack(y, axis=0)
    adj_in_array = np.stack(adj_in, axis=0)
    adj_out_array = np.stack(adj_out, axis=0)
    l1_array = np.stack(l1, axis=0)
    l2_array = np.stack(l2, axis=0)
    l3_array = np.stack(l3, axis=0)

    a, b, c, d = x_array.shape
    instances_count_on_vm = category_instance_df.values
    instances_count_on_vm = np.expand_dims(instances_count_on_vm, axis=0)
    instances_count_on_vm = np.expand_dims(instances_count_on_vm, axis=0)
    instances_count_on_vm_array = instances_count_on_vm.repeat(b, axis=1).repeat(
        a, axis=0
    )

    # set data type
    x_array = x_array.astype("float32")
    y_array = y_array.astype("float32")
    adj_in_array = adj_in_array.astype("float16")
    adj_out_array = adj_out_array.astype("float16")
    l1_array = l1_array.astype("int8")
    l2_array = l2_array.astype("int8")
    l3_array = l3_array.astype("int8")
    instances_count_on_vm_array = instances_count_on_vm_array.astype("int8")
    return (
        x_array,
        y_array,
        adj_in_array,
        adj_out_array,
        l1_array,
        l2_array,
        l3_array,
        instances_count_on_vm_array,
        arr_min,
        arr_max,
    )


def generate_graph_seq2seq_io_data(
    dataset_name, data, x_offsets, y_offsets, num_of_instances
):
    x, y = [], []

    add_time_in_day = True  # for METR-LA
    add_day_in_week = False  # for METR-LA

    if isinstance(data, np.ndarray):
        t, n, f = data.shape
    elif isinstance(data, pd.DataFrame):
        if dataset_name == "METR-LA":
            data.fillna(0, inplace=True)
            df = data
            data = np.expand_dims(df.values, axis=-1)  # T N 1
            t, n, f = data.shape
    if dataset_name == "METR-LA":
        data_list = [data]
        if add_time_in_day:
            time_ind = (
                df.index.values - df.index.values.astype("datetime64[D]")
            ) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, n, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if add_day_in_week:
            day_in_week = np.zeros(shape=(t, n, 7))
            day_in_week[np.arange(t), :, df.index.dayofweek] = 1
            data_list.append(day_in_week)
        data = np.concatenate(data_list, axis=-1)

    min_t = abs(min(x_offsets))  # -63--0  63
    max_t = abs(int(t) - abs(max(y_offsets)))  # 1--64  130-64 = 66
    for tau in range(min_t, max_t):
        x.append(data[(tau + x_offsets), :, :])
        y.append(data[(tau + y_offsets), :, :])
    x_array = np.stack(x, axis=0)
    y_array = np.stack(y, axis=0)

    return x_array, y_array


def shuffle_before_load_for_trainticket(
    x, y, adj_in, adj_out, l1, l2, l3, instances_count
):
    """Shuffle Dataset"""
    permutation = np.random.permutation(x.shape[0])
    xs, ys = x[permutation], y[permutation]
    ins, outs = adj_in[permutation], adj_out[permutation]
    l1s, l2s, l3s = l1[permutation], l2[permutation], l3[permutation]
    instances_counts = instances_count[permutation]
    return xs, ys, ins, outs, l1s, l2s, l3s, instances_counts


def shuffle_before_load(x, y):
    """Shuffle Dataset"""
    permutation = np.random.permutation(x.shape[0])
    xs, ys = x[permutation], y[permutation]
    return xs, ys


def generate_train_val_test_for_trainticket(args, dataset_path):
    """
    Generating train val test dataset for training.
    Reading dataset from processed dataset csv file.
    :param args:
    :param dataset_path: dataset path
    :return:
    """
    if sys.platform.startswith("win"):
        base_path = args.data_path
    elif sys.platform.startswith("linux"):
        base_path = "./"
    sample_rate_str = str(args.sample_rate)
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    category_list = scan_category(dataset_path)
    x_all = []
    y_all = []
    adj_in_all = []
    adj_out_all = []
    l1_all = []
    l2_all = []
    l3_all = []
    instances_count_all = []

    min_final = None
    max_final = None

    # 0 is the latest observed sample.  -15--0
    x_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1))
    # Predict the next step one hour     1--16
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    adj_in_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1))
    adj_out_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    #                   "io-latency200-20-1h-5s", "mem-stress30-15-1h-5s", "mem-stress30-20-1h-5s",
    #                   "network-corrupt15-15-2h-5s", "network-corrupt20-15-2h-5s", "network-delay200-15-2h-5s",
    #                   "network-delay200-20-2h-5s", "network-loss15-05-2h-5s", "network-loss15-15-2h-5s",
    #                   "network-loss15-25-2h-5s", "normal-05-2h-5s", "normal-15-2h-5s", "normal-25-2h-5s"]

    processed_list = []

    for category in tqdm.tqdm(category_list):
        if category in processed_list:
            continue
        if category.find("-" + sample_rate_str + "s") == -1:
            continue
        if sys.platform.startswith("win"):
            cat_short = category.split("\\")[-1]
            category_path = "./" + category + "/"
            category_metric_path = (
                category_path + category + "_all_nodes_metric_labeled_df" + ".csv"
            )
            category_instance_path = (
                category_path + category + "_instances_index_vm_df" + ".csv"
            )
            category_connection_path = (
                category_path + category + "_all_connection_df" + ".csv"
            )
        elif sys.platform.startswith("linux"):
            cat_short = category.split("/")[-1]
            category_path = category + "/"
            category_metric_path = (
                category_path + cat_short + "_all_nodes_metric_labeled_df" + ".csv"
            )
            category_instance_path = (
                category_path + cat_short + "_instances_index_vm_df" + ".csv"
            )
            category_connection_path = (
                category_path + cat_short + "_all_connection_df" + ".csv"
            )

        category_metric_df = pd.read_csv(category_metric_path, sep=",", header=0)
        category_instance_df = pd.read_csv(category_instance_path, sep=",", header=0)
        category_connection_df = pd.read_csv(
            category_connection_path, sep=",", header=0, index_col=0
        )

        category_metric_df = preprocess_df(category_metric_df)

        x, y, adj_in, adj_out, l1, l2, l3, instances_count, arr_min, arr_max = (
            generate_graph_seq2seq_io_data_for_trainticket(
                category_metric_df,
                category_instance_df,
                category_connection_df,
                x_offsets=x_offsets,
                y_offsets=y_offsets,
                adj_in_offsets=adj_in_offsets,
                adj_out_offsets=adj_out_offsets,
                sample_rate=sample_rate_str,
                num_of_instances=args.num_of_instances,
            )
        )

        if min_final is None:
            min_final = arr_min
            max_final = arr_max
        else:
            min_final, max_final = update_min_max(
                min_final, max_final, arr_min, arr_max
            )

        if args.if_shuffle:
            x, y, adj_in, adj_out, l1, l2, l3, instances_count = (
                shuffle_before_load_for_trainticket(
                    x, y, adj_in, adj_out, l1, l2, l3, instances_count
                )
            )

        # Save windowed arrays and metadata for the generic NPZ workflow.
        num_samples = x.shape[0]
        num_test = round(num_samples * 0.10)  # 10 % testing
        num_train = round(num_samples * 0.80)  # 80 % training
        num_val = num_samples - num_test - num_train  # 10 % validation

        x_train, y_train = x[:num_train], y[:num_train]
        adj_in_train, adj_out_train = adj_in[:num_train], adj_out[:num_train]
        l1_train, l2_train, l3_train = l1[:num_train], l2[:num_train], l3[:num_train]
        instances_count_train = instances_count[:num_train]

        x_val, y_val = (
            x[num_train : num_train + num_val],
            y[num_train : num_train + num_val],
        )
        adj_in_val, adj_out_val = (
            adj_in[num_train : num_train + num_val],
            adj_out[num_train : num_train + num_val],
        )
        l1_val, l2_val, l3_val = (
            l1[num_train : num_train + num_val],
            l2[num_train : num_train + num_val],
            l3[num_train : num_train + num_val],
        )
        instances_count_val = instances_count[num_train : num_train + num_val]

        x_test, y_test = x[-num_test:], y[-num_test:]
        adj_in_test, adj_out_test = adj_in[-num_test:], adj_out[-num_test:]
        l1_test, l2_test, l3_test = l1[-num_test:], l2[-num_test:], l3[-num_test:]
        instances_count_test = instances_count[-num_test:]

        for cat in ["train", "val", "test"]:
            # Use local variables to keep the chunk schema explicit.
            _x, _y = locals()["x_" + cat], locals()["y_" + cat]
            _adj_in, _adj_out = locals()["adj_in_" + cat], locals()["adj_out_" + cat]
            _l1, _l2, _l3 = (
                locals()["l1_" + cat],
                locals()["l2_" + cat],
                locals()["l3_" + cat],
            )
            _instances_count = locals()["instances_count_" + cat]
            print("Beginning to save {}_{} dataset".format(cat_short, cat))
            print(cat, "x: ", _x.shape, "y:", _y.shape)
            print(cat, "adj_in: ", _adj_in.shape, "adj_out:", _adj_out.shape)
            print(cat, "l1: ", _l1.shape, "l2:", _l2.shape, "l3:", _l3.shape)
            print(cat, "instances_count: ", _instances_count.shape)
            if sys.platform.startswith("win"):
                npz_path = "./" + os.path.join(
                    category, "{}_{}_{}.npz".format(cat_short, cat, sample_rate_str)
                )
            elif sys.platform.startswith("linux"):
                npz_path = os.path.join(
                    category, "{}_{}_{}.npz".format(cat_short, cat, sample_rate_str)
                )

            np.savez(
                npz_path,
                x=_x,
                y=_y,
                adj_in=_adj_in,
                adj_out=_adj_out,
                l1=_l1,
                l2=_l2,
                l3=_l3,
                instances_count=_instances_count,
            )
            print(
                "{}_{} dataset has been processed and saved successfully!".format(
                    cat_short, cat
                )
            )
    print("-" * 100)
    print("All category dataset has been processed and saved successfully!")


def get_adjacency_matrix_for_PEMS(file, num_of_vertices):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    # Distance-file node IDs start from 0.
    with open(file, "r") as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            A[i, j] = 1
            A[j, i] = 1
            distanceA[i, j] = distance
            distanceA[j, i] = distance
    return A, distanceA


def get_adjacency_matrix_direction_for_PEMS(file, num_of_vertices):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    # Distance-file node IDs start from 0.
    with open(file, "r") as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            A[i, j] = 1
            distanceA[i, j] = distance
    return A, distanceA


def generate_train_val_test(args, dataset_path, dataset):
    """
    Generating train val test dataset for training.
    Reading dataset from processed dataset csv file.
    :param args:
    :param dataset_path: dataset path
    :param dataset: dataset name
    :return:
    """
    global adj_mx, distance_mx, x, y, scaler
    dataset_name = dataset
    base_path = os.getcwd()
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    direction = args.if_direction
    min_final = None
    max_final = None

    # 0 is the latest observed sample.  -15--0
    x_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1))
    # Predict the next step one hour     1--16
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    if dataset_name == "trainticket":
        sample_rate_str = str(args.sample_rate)
        category_list = scan_category(dataset_path)
        adj_in_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1))
        adj_out_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

        processed_list = []

        for category in tqdm.tqdm(category_list):
            if category in processed_list:
                continue
            if category.find("-" + sample_rate_str + "s") == -1:
                continue
            if sys.platform.startswith("win"):
                cat_short = category.split("\\")[-1]
            elif sys.platform.startswith("linux"):
                cat_short = category.split("/")[-1]
            category_path = os.path.join(os.getcwd(), category)
            category_metric_path = os.path.join(
                category_path, cat_short + "_all_nodes_metric_labeled_df" + ".csv"
            )
            category_instance_path = os.path.join(
                category_path, cat_short + "_instances_index_vm_df" + ".csv"
            )
            category_connection_path = os.path.join(
                category_path, cat_short + "_all_connection_df" + ".csv"
            )

            category_metric_df = pd.read_csv(category_metric_path, sep=",", header=0)
            category_instance_df = pd.read_csv(
                category_instance_path, sep=",", header=0
            )
            category_connection_df = pd.read_csv(
                category_connection_path, sep=",", header=0, index_col=0
            )

            category_metric_df = preprocess_df(category_metric_df)

            x, y, adj_in, adj_out, l1, l2, l3, instances_count, arr_min, arr_max = (
                generate_graph_seq2seq_io_data_for_trainticket(
                    category_metric_df,
                    category_instance_df,
                    category_connection_df,
                    x_offsets=x_offsets,
                    y_offsets=y_offsets,
                    adj_in_offsets=adj_in_offsets,
                    adj_out_offsets=adj_out_offsets,
                    sample_rate=sample_rate_str,
                    num_of_instances=args.num_of_instances,
                )
            )

            if min_final is None:
                min_final = arr_min
                max_final = arr_max
            else:
                min_final, max_final = update_min_max(
                    min_final, max_final, arr_min, arr_max
                )

            if args.if_shuffle:
                x, y, adj_in, adj_out, l1, l2, l3, instances_count = (
                    shuffle_before_load_for_trainticket(
                        x, y, adj_in, adj_out, l1, l2, l3, instances_count
                    )
                )

            # Save the generated split as NPZ.
            num_samples = x.shape[0]
            num_test = round(num_samples * 0.10)  # 10 % testing
            num_train = round(num_samples * 0.80)  # 80 % training
            num_val = num_samples - num_test - num_train  # 10 % validation

            x_train, y_train = x[:num_train], y[:num_train]
            adj_in_train, adj_out_train = adj_in[:num_train], adj_out[:num_train]
            l1_train, l2_train, l3_train = (
                l1[:num_train],
                l2[:num_train],
                l3[:num_train],
            )
            instances_count_train = instances_count[:num_train]

            x_val, y_val = (
                x[num_train : num_train + num_val],
                y[num_train : num_train + num_val],
            )
            adj_in_val, adj_out_val = (
                adj_in[num_train : num_train + num_val],
                adj_out[num_train : num_train + num_val],
            )
            l1_val, l2_val, l3_val = (
                l1[num_train : num_train + num_val],
                l2[num_train : num_train + num_val],
                l3[num_train : num_train + num_val],
            )
            instances_count_val = instances_count[num_train : num_train + num_val]

            x_test, y_test = x[-num_test:], y[-num_test:]
            adj_in_test, adj_out_test = adj_in[-num_test:], adj_out[-num_test:]
            l1_test, l2_test, l3_test = l1[-num_test:], l2[-num_test:], l3[-num_test:]
            instances_count_test = instances_count[-num_test:]

            for cat in ["train", "val", "test"]:
                # Use local variables to keep the chunk schema explicit.
                _x, _y = locals()["x_" + cat], locals()["y_" + cat]
                _adj_in, _adj_out = (
                    locals()["adj_in_" + cat],
                    locals()["adj_out_" + cat],
                )
                _l1, _l2, _l3 = (
                    locals()["l1_" + cat],
                    locals()["l2_" + cat],
                    locals()["l3_" + cat],
                )
                _instances_count = locals()["instances_count_" + cat]
                print("Beginning to save {}_{} dataset".format(cat_short, cat))
                print(cat, "x: ", _x.shape, "y:", _y.shape)
                print(cat, "adj_in: ", _adj_in.shape, "adj_out:", _adj_out.shape)
                print(cat, "l1: ", _l1.shape, "l2:", _l2.shape, "l3:", _l3.shape)
                print(cat, "instances_count: ", _instances_count.shape)
                npz_path = os.path.join(
                    dataset_path,
                    category,
                    "{}_{}_{}.npz".format(cat_short, cat, sample_rate_str),
                )

                np.savez(
                    npz_path,
                    x=_x,
                    y=_y,
                    adj_in=_adj_in,
                    adj_out=_adj_out,
                    l1=_l1,
                    l2=_l2,
                    l3=_l3,
                    instances_count=_instances_count,
                )
                print(
                    "{}_{} dataset has been processed and saved successfully!".format(
                        cat_short, cat
                    )
                )

        scaler_npz_path = os.path.join(
            dataset_path, "{}_{}.npz".format(dataset_name, "scaler")
        )
        np.savez(scaler_npz_path, min=min_final, max=max_final)
    else:
        files, _ = get_subfiles_subfolders(dataset_path)
        if dataset_name in ["electricity", "exchange_rate", "solar-energy", "traffic"]:
            for file in files:
                if file.endswith("txt"):
                    f = open(file)
                    raw_data = np.loadtxt(f, delimiter=",")
                    print("input data size:" + str(raw_data.shape))
                    t, n = raw_data.shape
                    raw_data = raw_data.reshape(t, n, 1)
                    data, scaler = normalize_dataset(raw_data, normalize=2)
                    x, y = generate_graph_seq2seq_io_data(
                        dataset_name, data, x_offsets, y_offsets, n
                    )
                    if args.if_shuffle:
                        x, y = shuffle_before_load(x, y)
        elif dataset_name == "AIR-BJ":
            for file in files:
                if file.endswith("BJ.npz"):
                    raw_data = np.load(file)["arr"]
                    t, n, f = raw_data.shape
                    # data, scaler = standard_transform(raw_data)
                    data, scaler = normalize_dataset(raw_data, normalize=2)

                    x, y = generate_graph_seq2seq_io_data(
                        dataset_name, data, x_offsets, y_offsets, n
                    )
                    if args.if_shuffle:
                        x, y = shuffle_before_load(x, y)
        elif dataset_name == "METR-LA":
            for file in files:
                if file.endswith("h5"):
                    # pip install tables
                    raw_data = pd.read_hdf(file)
                    raw_data.index = pd.to_datetime(raw_data.index.astype(str))
                    print("input data size:" + str(raw_data.shape))
                    t, n = raw_data.shape
                    raw_data_value = raw_data.values.reshape(t, n, 1)
                    new_data, scaler = normalize_dataset(raw_data_value, normalize=2)
                    new_df = pd.DataFrame(
                        data=new_data.reshape(t, n),
                        index=raw_data.index,
                        columns=raw_data.columns,
                    )
                    x, y = generate_graph_seq2seq_io_data(
                        dataset_name, new_df, x_offsets, y_offsets, n
                    )
                    if args.if_shuffle:
                        x, y = shuffle_before_load(x, y)
        elif dataset_name in ["PEMS04", "PEMS08"]:
            if dataset_name == "PEMS04":
                num_nodes = 307
            elif dataset_name == "PEMS08":
                num_nodes = 170
            for file in files:
                if file.endswith("04.npz") or file.endswith("08.npz"):
                    raw_data = np.load(file)[
                        "data"
                    ]  # PEMS04 (16992, 307, 3), PEMS08 (17856, 170, 3)
                    t, n, f = raw_data.shape
                    data, scaler = standard_transform(raw_data)
                    x, y = generate_graph_seq2seq_io_data(
                        dataset_name, data, x_offsets, y_offsets, n
                    )
                    if args.if_shuffle:
                        x, y = shuffle_before_load(x, y)
                elif file.endswith("csv"):
                    if direction:
                        adj_mx, distance_mx = get_adjacency_matrix_direction_for_PEMS(
                            file, num_nodes
                        )
                    else:
                        adj_mx, distance_mx = get_adjacency_matrix_for_PEMS(
                            file, num_nodes
                        )
                    add_self_loop = False
                    if add_self_loop:
                        adj_mx = adj_mx + np.identity(adj_mx.shape[0])
                        distance_mx = distance_mx + np.identity(distance_mx.shape[0])

        # Save windowed arrays and metadata for the generic NPZ workflow.
        num_samples = x.shape[0]
        num_test = round(num_samples * 0.10)  # 10 % testing
        num_train = round(num_samples * 0.80)  # 80 % training
        num_val = num_samples - num_test - num_train  # 10 % validation

        x_train, y_train = x[:num_train], y[:num_train]

        x_val, y_val = (
            x[num_train : num_train + num_val],
            y[num_train : num_train + num_val],
        )

        x_test, y_test = x[-num_test:], y[-num_test:]

        for cat in ["train", "val", "test"]:
            # Use local variables to keep the chunk schema explicit.
            _x, _y = locals()["x_" + cat], locals()["y_" + cat]
            print("Beginning to save {}_{} dataset".format(dataset_name, cat))
            print(cat, "x: ", _x.shape, "y:", _y.shape)
            npz_path = os.path.join(dataset_path, "{}_{}.npz".format(dataset_name, cat))
            if dataset_name in ["PEMS04", "PEMS08"]:
                np.savez(
                    npz_path,
                    x=_x,
                    y=_y,
                    adj=adj_mx,
                    adj_d=distance_mx,
                )
            else:
                np.savez(
                    npz_path,
                    x=_x,
                    y=_y,
                )
            print(
                "{}_{} dataset has been processed and saved successfully!".format(
                    dataset_name, cat
                )
            )

        scaler_npz_path = os.path.join(
            dataset_path, "{}_{}.npz".format(dataset_name, "scaler")
        )
        np.savez(
            scaler_npz_path,
            scaler=scaler,
        )


if __name__ == "__main__":
    """
    {0: 'Germany Dataset', 1: 'Hades Dataset', 2: 'OB-Nezha Dataset', 3: 'SN-Eadro Dataset',
    4: 'TT-Nezha Dataset', 5: 'TT-Ebpf Dataset', 6: 'TT- Dataset'}
    """
    parser = argparse.ArgumentParser()

    # designed to generate dataset on Windows
    parser.add_argument(
        "--data_path",
        type=str,
        default="./",
        help="Raw Data.",
    )
    parser.add_argument(
        "--dataset_index",
        type=int,
        default=7,
        help="Dataset index.",
    )
    parser.add_argument(
        "--if_process_all",
        type=eval,
        default=True,
        help="Whether to process all raw datasets",
    )
    parser.add_argument(
        "--seq_length_x",
        type=int,
        default=16,
        help="Input Sequence Length.",
    )
    parser.add_argument(
        "--seq_length_y",
        type=int,
        default=16,
        help="Output Sequence Length.",
    )
    parser.add_argument(
        "--num_of_instances",
        type=int,
        default=56,
        help="Nodes number",
    )
    parser.add_argument(
        "--if_direction",
        type=eval,
        default=False,
        help="Whether generate directed graph",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=5,
        help="the sample rate of collecting metrics",
    )
    parser.add_argument(
        "--if_shuffle",
        type=eval,
        default=False,
        help="Whether to shuffle the dataset",
    )
    parser.add_argument(
        "--if_scale", type=eval, default=True, help="Whether to scale the raw dataset"
    )
    parser.add_argument(
        "--y_start",
        type=int,
        default=1,
        help="Y pred start",
    )
    parser.add_argument(
        "--type_of_adj",
        type=str,
        default="counts",
        help="choose the type of adj",
    )

    args = parser.parse_args()
    _, dataset_list = get_subfiles_subfolders(args.data_path)
    idx_to_dataset = {}
    for i in range(len(dataset_list)):
        if sys.platform.startswith("win"):
            dataset_name = dataset_list[i].split("\\")[-1]
        elif sys.platform.startswith("linux"):
            dataset_name = dataset_list[i].split("/")[-1]
        if dataset_name == "__pycache__":
            del dataset_list[i]
            continue
        idx_to_dataset[i] = dataset_name
    if args.if_process_all:
        for k, v in tqdm.tqdm(idx_to_dataset.items()):
            print(f"Beginning to process {v}...")
            generate_train_val_test(args, dataset_list[k], v)
            print(f"{v} has been processed successfully!")
        print("All datasets have been processed successfully!")
    else:
        dataset_chosen = idx_to_dataset[args.dataset_index]
        print(f"Beginning to process {dataset_chosen}...")
        generate_train_val_test(args, dataset_list[args.dataset_index], dataset_chosen)
        print(f"{dataset_chosen} has been processed successfully!")

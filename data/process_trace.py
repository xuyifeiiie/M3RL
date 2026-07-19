"""
Preprocess traces and dynamic dependency features for M3RL.

Trace spans provide the time-varying service dependency graph used by STNet.
For each sliding window, this module extracts service-to-service invocations,
latency features, dynamic adjacency matrices, and trace-derived time-series
statistics. These outputs are aligned with metric and log windows by
`align_multimodal_main.py`.
"""

import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import pandas as pd
import numpy as np
import csv
import re
import json
import time
import pickle
from tqdm import tqdm

from data_utils import read_json


TRACERESULT = []


def get_parent_service_name_eadro(parent_span_id, spans, names):
    for span in spans:
        if span["spanID"] != parent_span_id:
            continue
        return get_service_name_eadro(span, names)


def get_service_name_eadro(span, names):
    if span["processID"]:
        return names[span["processID"]]
    return ""


def find_call_eadro(span, spans, names):
    """Return the caller/callee service pair represented by an Eadro span."""
    if len(span["references"]) == 0:
        serviceName = get_service_name_eadro(span, names)
        return True, serviceName, serviceName
    if len(span["references"]) > 0 and span["references"][0]["refType"] == "CHILD_OF":
        serviceName = get_service_name_eadro(span, names)
        parentServiceName = get_parent_service_name_eadro(
            span["references"][0]["spanID"], spans, names
        )
        return True, parentServiceName, serviceName
    return False, "", ""


def read_trace_germany(data, base_trace=None, name="start", isfirst=False):
    """Flatten nested Germany traces into span rows."""
    if isfirst:
        pass
    else:
        span_id = data["trace_id"]  # Span identifier in the Germany trace format.
        parentspan_id = data["parent_id"]
        info = data["info"]
        cmbd_id = info["host"]
        spantype = info["name"]
        try:
            # Germany traces store start/stop event names as dynamic keys.
            start_event = list(
                filter(
                    lambda x: any("-start" in x for _ in list(info.keys())),
                    list(info.keys()),
                )
            )[0]
            span_start_time = info[start_event]["timestamp"]
            end_event = list(
                filter(
                    lambda x: any("-stop" in x for _ in list(info.keys())),
                    list(info.keys()),
                )
            )[0]
            span_end_time = info[end_event]["timestamp"]
        except:
            span_end_time = "wrong"
        TRACERESULT.append(
            [
                cmbd_id,
                span_start_time,
                span_end_time,
                spantype,
                span_id,
                parentspan_id,
                base_trace,
                name,
            ]
        )
        name = cmbd_id
    # Recursively traverse child spans.
    if data["children"]:
        for span in data["children"]:
            read_trace_germany(span, base_trace, name)


def deal_ground_germany(datapath):
    result = []
    for name in os.listdir(datapath):
        namel = re.split(r"[_.]", name)
        if namel[-1] != "json":
            continue

        # Read one trace report JSON file.
        with open(os.path.join(datapath, name)) as f:
            data = json.load(f)  # Output: dict

        data = data["tasks"][0]["subtasks"][0]["workloads"][0]["data"]
        print(os.path.join(datapath, name), "report data:", len(data))
        for item in data:
            timesstamp = item["timestamp"]
            error = 1 if item["error"] else 0
            duration = item["duration"]
            traceid = item["output"]["complete"][0]["data"]["trace_id"]
            action = (
                item["atomic_actions"][-1]["name"],
                item["atomic_actions"][-1]["started_at"],
                item["atomic_actions"][-1]["finished_at"],
                1 if "failed" in list(item["atomic_actions"][-1].keys()) else 0,
            )
            result.append([traceid, timesstamp, duration, error, *action])

    # Keep trace-level report fields used to derive anomaly intervals.
    ground = pd.DataFrame(
        result,
        columns=[
            "traceid",
            "timestamp",
            "duration",
            "error",
            "ac_name",
            "ac_start",
            "ac_end",
            "ac_status",
        ],
    )
    return ground


def process_trace(dataset_info):
    """Convert raw spans into processed trace files for each dataset split."""
    dataset_name = dataset_info.dataset_name
    trace_data_path = dataset_info.metadata["data_path"]["traces"]
    data_processed_path = os.path.join(dataset_info.dataset_path, "processed")
    service_list = dataset_info.service_names
    if not os.path.exists(data_processed_path):
        os.makedirs(data_processed_path)

    if dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        for key, value in trace_data_path.items():
            # Eadro spans are grouped by fault/faultless experiment folders.
            if key == "fault":
                trace_processed_dir = os.path.join(
                    data_processed_path, "fault", "traces"
                )
                if not os.path.exists(trace_processed_dir):
                    os.makedirs(trace_processed_dir)
            else:
                trace_processed_dir = os.path.join(
                    data_processed_path, "faultless", "traces"
                )
                if not os.path.exists(trace_processed_dir):
                    os.makedirs(trace_processed_dir)

            for trace_path in value:
                if sys.platform.startswith("win"):
                    subcat = trace_path.split("\\")[-2]
                else:
                    subcat = trace_path.split("/")[-2]
                with open(trace_path, "rb") as f:
                    data = json.load(f)
                    result = {}
                    # Each item is one trace with multiple spans.
                    for item in data:
                        names = {}
                        processes = item["processes"]
                        for process in processes:
                            names[process] = processes[process]["serviceName"]
                        spans = item["spans"]
                        for span in spans:
                            is_call = False
                            is_call, parent_service_name, service_name = (
                                find_call_eadro(span, spans, names)
                            )
                            if not is_call:
                                continue
                            curTime = str(int(span["startTime"] / 1e6) - 3600 * 8)
                            if parent_service_name is None:
                                continue
                            call = "%d-%d" % (
                                dataset_info.service2id[parent_service_name],
                                dataset_info.service2id[service_name],
                            )
                            if curTime not in result:
                                result[curTime] = {}
                            if call not in result[curTime]:
                                result[curTime][call] = []
                            result[curTime][call].append(span["duration"])
                output_file = os.path.join(trace_processed_dir, subcat + "_traces.json")
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)

    elif dataset_name in ["Germany Dataset"]:
        for key, value in trace_data_path.items():
            # Germany spans are grouped by concurrent/sequential experiment folders.
            if key == "con":
                flag = "concurrent"
                flag_full = "concurrent data"
                start = dataset_info.metadata[
                    "concurrent data_IMPORTANT_experiment_start_end_data"
                ]["trace_start"]
                end = dataset_info.metadata[
                    "concurrent data_IMPORTANT_experiment_start_end_data"
                ]["trace_end"]
                trace_processed_dir = os.path.join(
                    data_processed_path, flag_full, "traces"
                )
                if not os.path.exists(trace_processed_dir):
                    os.makedirs(trace_processed_dir)
            else:
                flag = "sequential"
                flag_full = "sequential_data"
                start = dataset_info.metadata[
                    "sequential_data_IMPORTANT_experiment_start_end_data"
                ]["trace_start"]
                end = dataset_info.metadata[
                    "sequential_data_IMPORTANT_experiment_start_end_data"
                ]["trace_end"]
                trace_processed_dir = os.path.join(
                    data_processed_path, flag_full, "traces"
                )
                if not os.path.exists(trace_processed_dir):
                    os.makedirs(trace_processed_dir)

            processed_cat_raw_file = os.path.join(trace_processed_dir, "raw_trace.csv")
            if os.path.exists(processed_cat_raw_file):
                trace = pd.read_csv(processed_cat_raw_file)
            else:
                for file_path in value:
                    data = json.load(open(file_path))
                    name = os.path.basename(file_path).split(".")[0]
                    read_trace_germany(
                        data, base_trace=name, name="start", isfirst=True
                    )

                trace = pd.DataFrame(
                    TRACERESULT,
                    columns=[
                        "cmbd_id",
                        "start_time",
                        "end_time",
                        "stats",
                        "span_id",
                        "parentspan_id",
                        "base_trace",
                        "fatherpod",
                    ],
                )
                trace.to_csv(
                    os.path.join(trace_processed_dir, "raw_trace.csv"), index=False
                )

            processed_cat_file = os.path.join(trace_processed_dir, flag + "_traces.csv")
            if os.path.exists(processed_cat_file):
                trace = pd.read_csv(processed_cat_file)
            else:
                start_time_values = trace["start_time"].values
                split_results = [s.split(".", 1) for s in start_time_values]
                start_time_df = pd.DataFrame(
                    split_results, columns=["start_time", "start_sec"]
                )
                trace[["start_time", "start_sec"]] = start_time_df
                trace["start_time"] = trace["start_time"].map(
                    lambda x: time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%S"))
                )

                trace["end_time"].replace("wrong", np.nan, inplace=True)
                trace["end_time"].fillna(method="ffill", inplace=True)

                end_time_values = trace["end_time"].values
                split_results = [s.split(".", 1) for s in end_time_values]
                end_time_df = pd.DataFrame(
                    split_results, columns=["end_time", "end_sec"]
                )
                trace[["end_time", "end_sec"]] = end_time_df
                trace["end_time"] = trace["end_time"].map(
                    lambda x: time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%S"))
                )

                max_start_time = trace["start_time"].max()
                max_end_time = trace["end_time"].max()
                min_start_time = trace["start_time"].min()
                min_end_time = trace["end_time"].min()
                print(
                    "### Max start time: ", max_start_time
                )  # 1574681789.0 2019-11-25 19:36:29
                print(
                    "### Min start time: ", min_start_time
                )  # 1574665946.0 2019-11-25 15:12:26
                print("### Max end time: ", max_end_time)  # 1574681789.0
                print("### Min end time: ", min_end_time)  # 1574665946.0

                print("### Trace shape before processing: ", trace.shape)
                print("### Save trace data...")
                trace[["end_time", "start_time", "end_sec", "start_sec"]] = trace[
                    ["end_time", "start_time", "end_sec", "start_sec"]
                ].apply(pd.to_numeric)
                trace = trace.loc[
                    (trace["end_time"] >= start) & (trace["end_time"] <= end), :
                ]
                trace["duration"] = (
                    trace["end_time"]
                    - trace["start_time"]
                    + (trace["end_sec"] - trace["start_sec"]) * 1e-6
                )
                trace.to_csv(
                    os.path.join(trace_processed_dir, flag + "_traces.csv"), index=False
                )
                print("### Triming trace data by date:", trace.shape)  # (1514358, 11)

            print("### Dealing relation...")
            pod_relation = trace.apply(
                lambda x: "_".join([x["fatherpod"], x["cmbd_id"]]), axis=1
            )
            pod_relation = list(set(pod_relation))  # Deduplicate service pairs.

            # Static service adjacency captures observed communication pairs.
            relation_matrix = np.zeros((len(service_list), len(service_list)))
            for item in pod_relation:
                [start_pod, end_pod] = item.split("_")
                if start_pod not in service_list or end_pod not in service_list:
                    continue
                relation_matrix[
                    service_list.index(start_pod), service_list.index(end_pod)
                ] = 1
                relation_matrix[
                    service_list.index(end_pod), service_list.index(start_pod)
                ] = 1
            pickle.dump(
                relation_matrix,
                open(
                    os.path.join(trace_processed_dir, flag + "_static_relation.pkl"),
                    "wb",
                ),
            )

            # Extract report-level labels for the Germany trace subset.
            ground = deal_ground_germany(
                os.path.join(dataset_info.dataset_path, flag_full, "reports")
            )
            ground.to_csv(
                os.path.join(trace_processed_dir, flag + "_groundtruth.csv"),
                index=False,
            )
            print("### The ground shape is :", ground.shape)  # (11000, 8)
            # Keep error traces for anomaly interval construction.
            ground = ground.loc[ground["error"] == 1]
            ground[["ac_start", "ac_end"]] = ground[["ac_start", "ac_end"]].apply(
                pd.to_numeric
            )
            min_start_ground_time = ground["ac_start"].min()
            max_start_ground_time = ground["ac_start"].max()
            min_end_ground_time = ground["ac_end"].min()
            max_end_ground_time = ground["ac_end"].max()
            print("### The min_start_ground_time is :", min_start_ground_time)  # 0
            print("### The max_start_ground_time is :", max_start_ground_time)  # 11160
            print("### The min_end_ground_time is :", min_end_ground_time)  # 0
            print("### The max_end_ground_time is :", max_end_ground_time)  # 11160
            ground_labels = np.zeros(
                (int((end - start) + 1), len(service_list))
            )  # label (11161,5)
            print(
                "### Error ground shape is :", ground.shape
            )  # Error-trace ground shape example: (2636, 8).

            for _, item in ground.iterrows():
                trace_data = trace.loc[
                    trace["base_trace"] == item["traceid"]
                ]  # Match reports to corresponding trace IDs.
                cmbd_list = trace_data["cmbd_id"].unique().tolist()
                cmbd_index = list(map(lambda x: service_list.index(x), cmbd_list))
                if int(item["ac_end"] - start + 1) >= int((end - start) + 1):
                    continue
                else:
                    time_list = [
                        item
                        for item in range(
                            int(item["ac_start"] - start),
                            int(item["ac_end"] - start + 1),
                        )
                    ]
                    for x in cmbd_index:
                        ground_labels[time_list, x] = 1

            pickle.dump(
                ground_labels,
                open(
                    os.path.join(trace_processed_dir, flag + "_ground_labels.pkl"), "wb"
                ),
            )


z_zero_scaler = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8)


def deal_traces(x_intervals, y_intervals, info, cat, subdir, params):
    """
    Input:
        intervals=[(s,e)], the chunks covers the period of [s, e].
    Return:
        a dict containing info for each interval:
        -- cell of invok list : a dict contains invocations inside the given time period == as invocation-based edge-level features
            {s-t:[lat1, lat2, ...]}
        -- cell of latency list: a dict contains a np.array [chunk_lenth] denoting the average latency (per time slot) for each node
                                === as trace-based node-level features
            {nid:np.array([lat_1, ..., lat_tau, ..., lat_chunk_lenth}])}
    """
    print(
        "***No traces.pkl file of {} founded, beginning to deal with preprocess traces...".format(
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
    threshold = params["threshold"]
    service_list = info.service_names
    dataset_name = info.dataset_name

    # Initialize invocation counts, latency features, and adjacency matrices.
    invocations = [[{} for _ in range(history_steps)] for _ in range(len(x_intervals))]
    latency = np.zeros(
        (len(x_intervals), info.node_num, history_steps, 1), dtype=np.float32
    )
    adj_matrix = np.zeros(
        (len(x_intervals), history_steps, info.node_num, info.node_num),
        dtype=np.float32,
    )
    time_series = np.zeros(
        (len(x_intervals), history_steps, info.node_num, 3), dtype=np.float32
    )

    print("*** Dealing with traces of {}...".format(subdir))
    # slots: 2022-04-17 10:12:45, traces: 2022-04-17 18:35:03
    if dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        traces = read_json(
            os.path.join(processed_path, cat, "traces", subdir + "_traces.json")
        )
        for chunk_idx, (s, e) in tqdm(enumerate(x_intervals)):
            slots = [t for t in range(s, e + 1)]
            for i, ts in enumerate(slots):
                if str(ts) in traces.keys():
                    spans = traces[str(ts)]
                    tmp_node_lat = [[] for _ in range(info.node_num)]
                    for k, lat_list in spans.items():
                        src, dst = map(int, k.split("-"))
                        invocations[chunk_idx][i][k] = len(lat_list)
                        tmp_node_lat[dst].extend(lat_list)
                    for t_node in range(info.node_num):
                        if len(tmp_node_lat[t_node]) > 0:
                            latency[chunk_idx][t_node][i][0] = np.mean(
                                tmp_node_lat[t_node]
                            )

        for i in range(info.node_num):
            latency[:, i, :, 0] = z_zero_scaler(latency[:, i, :, 0])

    elif dataset_name == "Germany Dataset":
        traces = pd.read_csv(
            os.path.join(processed_path, cat, "traces", subdir + "_traces.csv")
        )
        for chunk_idx, (s, e) in tqdm(enumerate(x_intervals)):
            slots = [t for t in range(s, e + 1)]
            for i, ts in enumerate(slots):
                trace_data = traces.loc[traces["start_time"] == ts]

                if not trace_data.empty:
                    tmp_node_lat = [[] for _ in range(info.node_num)]
                    for _, trace in trace_data.iterrows():
                        cmbd_id = trace["cmbd_id"]
                        span_id = trace["span_id"]
                        fatherpod = trace["fatherpod"]
                        duration = trace["duration"]
                        if fatherpod == "start":
                            continue
                        src = service_list.index(fatherpod)
                        dst = service_list.index(cmbd_id)
                        call_key = f"{src}-{dst}"
                        invocations[chunk_idx][i][call_key] = (
                            invocations[chunk_idx][i].get(call_key, 0) + 1
                        )

                        t_node = dst
                        tmp_node_lat[t_node].append(duration)

                    for t_node in range(info.node_num):
                        if tmp_node_lat[t_node]:
                            latency[chunk_idx][t_node][i][0] = np.mean(
                                tmp_node_lat[t_node]
                            )

        for i in range(info.node_num):
            latency[:, i, :, 0] = z_zero_scaler(latency[:, i, :, 0])

    # Build dynamic adjacency and trace time-series features for each window.
    for chunk_idx in range(len(x_intervals)):
        for t in range(history_steps):
            invok_t = invocations[chunk_idx][t]
            for call_key, call_count in invok_t.items():
                src, dst = map(int, call_key.split("-"))
                adj_matrix[chunk_idx, t, src, dst] = call_count

            for node in range(info.node_num):
                time_series[chunk_idx, t, node, 0] = latency[chunk_idx, node, t, 0]

                time_series[chunk_idx, t, node, 1] = np.sum(
                    adj_matrix[chunk_idx, t, node, :]
                )

                time_series[chunk_idx, t, node, 2] = np.sum(
                    adj_matrix[chunk_idx, t, :, node]
                )

    # Build a static adjacency matrix from all observed dynamic edges.
    static_adj_matrix = np.zeros((info.node_num, info.node_num), dtype=int)
    # Iterate over dynamic adjacency: len(x_intervals), history_steps, node_num.
    for s in range(len(x_intervals)):
        for t in range(history_steps):
            for i in range(info.node_num):
                for j in range(info.node_num):
                    if static_adj_matrix[i, j] == 1:
                        continue
                    else:
                        if adj_matrix[s, t, i, j] > 0:
                            static_adj_matrix[i, j] = 1

    # Persist trace-derived dynamic graphs and node-level trace features.
    chunk_traces = {
        "adj_matrix": adj_matrix,
        "metrics": time_series,
        "static_adj_matrix": static_adj_matrix,
    }
    with open(os.path.join(processed_path, cat, subdir + "-traces.pkl"), "wb") as fw:
        pickle.dump(chunk_traces, fw)

    return chunk_traces

"""
Preprocess raw logs for M3RL.

Logs are converted from raw service log files into structured templates and
per-window features. For the Eadro datasets used in the paper, raw JSON logs
are normalized to a common timestamp format, parsed by Drain, and later aligned
with metric/trace windows. Depending on `log_process_type`, downstream code can
consume either raw template sequences or template-count statistics.
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
import re

from datetime import datetime
from logparser import Drain
from utils.utils import get_subfiles_subfolders
from data_utils import read_json


def unique(l):
    return list(set(l))


def log_line_format_time_for_eadro(row):
    """Convert Eadro log timestamps to second-level Unix timestamps."""
    tmp = row.split(" ")
    dt = " ".join(tmp[0:2])
    if "[" in dt:
        dt = dt.replace("[", "")
    if "]" in dt:
        dt = dt.replace("]", "")

    try:
        ts = int(datetime.strptime(dt, "%Y-%b-%d %H:%M:%S.%f").timestamp())
    except ValueError:
        try:

            ts = int(datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f").timestamp())
        except ValueError:
            raise ValueError(f"Time data {dt} does not match any expected format.")
    return str(ts) + " " + " ".join(tmp[2:])


def parse_raw_log(dataset_name, log_file):
    lines = []
    if dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        with open(log_file, "rb") as fp:
            data = json.load(fp)
            for service, rows in data.items():
                for row in rows:
                    line = service + " " + log_line_format_time_for_eadro(row)
                    lines.append(line)
    elif dataset_name == "Germany Dataset":
        df = pd.read_csv(log_file, keep_default_na=False)
        data = df[["Hostname", "@timestamp", "log_level", "Payload"]]
        lines = [
            f"{row['Hostname']} {row['@timestamp']} {row['log_level']} {row['Payload']}"
            for _, row in data.iterrows()
        ]

    return lines


# Regexes remove request-specific tokens so Drain can learn stable templates.
instance_pattern = r"\[instance: [a-f0-9\-]+\]"
req_pattern = r"\[req-[a-f0-9\-]+\]"
stick_pattern = r"^- - - \[.*?\] "
timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \d+ ERROR [a-zA-Z\.]+"
url_pattern = (
    r'<url id="[^"]+" type="[^"]+" status="[^"]+" title="" wc="\d+">([^<]+)</url>'
)
apache_log_pattern1 = r"- - - \[\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}\]"
apache_log_pattern2 = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3} - - \[\d{2}/[A-Za-z]{3}/\d{4} \d{2}:\d{2}:\d{2}\]"
ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"


def clean_and_concatenate(log_text):
    """Remove request-specific patterns and collapse multi-line logs."""
    log_text = re.sub(url_pattern, r"\1", log_text)

    lines = log_text.split("\n")

    cleaned_lines = []
    for line in lines:
        line = line.strip()
        line = re.sub(instance_pattern, "", line).strip()
        line = re.sub(req_pattern, "", line).strip()
        line = re.sub(timestamp_pattern, "", line).strip()
        line = re.sub(apache_log_pattern1, "", line).strip()
        line = re.sub(apache_log_pattern2, "", line).strip()
        line = re.sub(ip_pattern, "", line).strip()
        cleaned_lines.append(line)

    concatenated_log = " ".join(cleaned_lines).strip()
    return concatenated_log


def detect_and_process(log_text):
    """Apply log cleaning only when known raw-log patterns are present."""
    if (
        re.search(timestamp_pattern, log_text)
        or re.search(apache_log_pattern1, log_text)
        or re.search(apache_log_pattern2, log_text)
    ):
        return clean_and_concatenate(log_text)
    else:
        return log_text


def process_log(dataset_info, parser_type="drain"):
    """Parse raw logs and save structured per-service log files."""
    dataset_name = dataset_info.dataset_name
    log_data_path = dataset_info.metadata["data_path"]["logs"]
    data_processed_path = os.path.join(dataset_info.dataset_path, "processed")
    if not os.path.exists(data_processed_path):
        os.makedirs(data_processed_path)

    if dataset_name in ["SN-Eadro Dataset", "TT-Eadro Dataset"]:
        for key, value in log_data_path.items():
            # Eadro logs are grouped by fault/faultless experiment folders.
            if key == "fault":
                log_processed_dir = os.path.join(data_processed_path, "fault", "logs")
                if not os.path.exists(log_processed_dir):
                    os.makedirs(log_processed_dir)
            else:
                log_processed_dir = os.path.join(
                    data_processed_path, "faultless", "logs"
                )
                if not os.path.exists(log_processed_dir):
                    os.makedirs(log_processed_dir)

            for log_path in value:
                lines = parse_raw_log(dataset_name, log_path)
                log_parsed_txt_path = log_path + ".parsed.txt"
                with open(log_parsed_txt_path, "w") as fp:
                    fp.write("\n".join(lines))
                if parser_type == "drain":
                    parser = Drain.LogParser(
                        log_format=dataset_info.log_template,
                        savePath=log_processed_dir,
                        depth=dataset_info.drain_depth,
                        st=dataset_info.drain_similarity_threshold,
                        rex=dataset_info.regex,
                    )
                    if sys.platform.startswith("win"):
                        parser.logName = log_path.split("\\")[-2]
                    else:
                        # os.path.join(self.savePath, self.logName + '_structured.csv')
                        parser.logName = log_path.split("/")[-2]
                    parser.parse(log_parsed_txt_path)
                elif parser_type == "llm":
                    pass

        events = []
        for cat in ["fault", "faultless"]:
            log_processed_dir = os.path.join(data_processed_path, cat, "logs")
            files, _ = get_subfiles_subfolders(log_processed_dir)

            for file in files:
                if file.endswith("_structured.csv"):
                    file_name = os.path.basename(file).split("_")[0]  # SN.2022...
                    output_file = os.path.join(
                        log_processed_dir, file_name + "_logs.csv"
                    )

                    df = pd.read_csv(file)
                    df_tmp = df[
                        [
                            "Service",
                            "Timestamp",
                            "EventId",
                            "Content",
                            "EventTemplate",
                            "ParameterList",
                        ]
                    ]
                    df_out = df_tmp.rename(columns={"EventId": "Event"})
                    df_out.to_csv(output_file)
                if file.endswith("templates.csv"):
                    df = pd.read_csv(file)
                    #     events.append(event)
                    for _, row in df.iterrows():
                        events.append(
                            {
                                "EventId": row["EventId"],
                                "EventTemplate": row["EventTemplate"],
                            }
                        )
        df_templates = pd.DataFrame(events)
        df_templates = df_templates.drop_duplicates(subset=["EventId"])
        template_output_file = os.path.join(data_processed_path, "template.csv")
        df_templates.to_csv(template_output_file, index=False)
        #     json.dump(unique(events), f, indent=2)

    elif dataset_name == "Germany Dataset":
        raw_log_path = ""
        for key, value in log_data_path.items():
            if key == "con":
                flag = "concurrent"
                start = dataset_info.metadata[
                    "concurrent data_IMPORTANT_experiment_start_end_data"
                ]["trace_start"]
                end = dataset_info.metadata[
                    "concurrent data_IMPORTANT_experiment_start_end_data"
                ]["trace_end"]
                log_processed_dir = os.path.join(
                    data_processed_path, flag + " data", "logs"
                )
                if not os.path.exists(log_processed_dir):
                    os.makedirs(log_processed_dir)
            else:
                flag = "sequential"
                start = dataset_info.metadata[
                    "sequential_data_IMPORTANT_experiment_start_end_data"
                ]["trace_start"]
                end = dataset_info.metadata[
                    "sequential_data_IMPORTANT_experiment_start_end_data"
                ]["trace_end"]
                log_processed_dir = os.path.join(
                    data_processed_path, flag + "_data", "logs"
                )
                if not os.path.exists(log_processed_dir):
                    os.makedirs(log_processed_dir)

            for tmp_path in value:
                if tmp_path.endswith("concurrent.csv") or tmp_path.endswith(
                    "sequential.csv"
                ):
                    raw_log_path = tmp_path
                    break
                else:
                    continue

            data = pd.read_csv(raw_log_path, keep_default_na=False)
            data = data[["Hostname", "@timestamp", "log_level", "Payload"]]
            data["log_level"].replace("", pd.NA, inplace=True)
            data["log_level"].fillna("INFO", inplace=True)
            data["Hostname"].replace("", pd.NA, inplace=True)
            data = data.dropna(subset=["Hostname"])
            data["@timestamp"] = data["@timestamp"].map(lambda x: x[:-10])
            data["@timestamp"] = data["@timestamp"].map(
                lambda x: time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"))
                - 60 * 60
            )
            data = data.loc[
                (data["@timestamp"] >= start) & (data["@timestamp"] <= end), :
            ]
            for pattern in [
                stick_pattern,
                instance_pattern,
                req_pattern,
                stick_pattern,
                timestamp_pattern,
                url_pattern,
                apache_log_pattern1,
                apache_log_pattern2,
                ip_pattern,
            ]:
                data["Payload"] = data["Payload"].str.replace(pattern, "", regex=True)
            # data['Payload'] = data['Payload'].apply(detect_and_process)
            data.to_csv(raw_log_path + "_trim.csv")
            # data.to_csv(raw_log_path + '_payload.csv')
            lines = parse_raw_log(dataset_name, raw_log_path + "_trim.csv")
            log_parsed_txt_path = (
                raw_log_path + "_parsed.txt"
            )  # # ../logs_aggregated_concurrent.csv_parsed.txt
            with open(log_parsed_txt_path, "w") as fp:
                fp.write("\n".join(lines))

            if parser_type == "drain":
                parser = Drain.LogParser(
                    log_format=dataset_info.log_template,
                    savePath=log_processed_dir,
                    depth=dataset_info.drain_depth,
                    st=dataset_info.drain_similarity_threshold,
                    rex=dataset_info.regex,
                )
                parser.logName = flag  # concurrent -> concurrent_structured.csv
                parser.parse(log_parsed_txt_path)
            elif parser_type == "llm":
                pass

        events = []
        for cat in ["concurrent data", "sequential_data"]:
            if cat == "concurrent data":
                flag = "concurrent"
            else:
                flag = "sequential"
            log_processed_dir = os.path.join(data_processed_path, cat, "logs")
            files, _ = get_subfiles_subfolders(log_processed_dir)

            for file in files:
                if file.endswith("_structured.csv"):
                    file_name = flag  # concurrent
                    output_file = os.path.join(
                        log_processed_dir, file_name + "_logs.csv"
                    )  # concurrent_logs.csv
                    df = pd.read_csv(file)
                    df_tmp = df[
                        [
                            "Service",
                            "Timestamp",
                            "EventId",
                            "Content",
                            "EventTemplate",
                            "ParameterList",
                        ]
                    ]
                    df_out = df_tmp.rename(columns={"EventId": "Event"})
                    float_pattern = r"^\d+\.\d+$"
                    mask = df_out["Timestamp"].astype(str).str.match(float_pattern)
                    df_out = df_out[mask]
                    df_out.to_csv(output_file)
                if file.endswith("templates.csv"):
                    for _, row in df.iterrows():
                        events.append(
                            {
                                "EventId": row["EventId"],
                                "EventTemplate": row["EventTemplate"],
                            }
                        )
        df_templates = pd.DataFrame(events)
        df_templates = df_templates.drop_duplicates(subset=["EventId"])
        template_output_file = os.path.join(data_processed_path, "template.csv")
        df_templates.to_csv(template_output_file, index=False)
        #                 events.append(event)

        #     json.dump(unique(events), f, indent=2)


def deal_logs(x_intervals, y_intervals, info, cat, subdir, params):
    print(
        "***No logs.pkl file of {} founded, beginning to deal with preprocess logs...".format(
            subdir
        )
    )
    log_process_type = params["log_process_type"]
    history_steps = params["history_steps"]
    predict_steps = params["predict_steps"]
    print("*** Dealing with logs of {}...".format(subdir))

    tmp_folder = os.getcwd()
    if tmp_folder.endswith("data"):
        datafolder = tmp_folder
    else:
        datafolder = os.path.join(tmp_folder, "data")
    processed_path = os.path.join(datafolder, info.dataset_name, "processed")

    df = pd.read_csv(os.path.join(processed_path, cat, "logs", subdir + "_logs.csv"))
    df = df[
        ["Timestamp", "Service", "Event", "Content", "EventTemplate", "ParameterList"]
    ]
    template_df = pd.read_csv(os.path.join(processed_path, "template.csv"))
    event_num = len(template_df)

    print("# Real Template Number:", event_num)
    event_num += 1  # add 1 to represent unseen template event, which idx is 0.
    event2id = {
        row["EventId"]: idx + 1 for idx, row in template_df.iterrows()
    }  # 0: unseen
    id2event_template = {
        idx + 1: row["EventTemplate"] for idx, row in template_df.iterrows()
    }
    event2id["Unseen"] = 0

    res_stats = np.zeros(
        (len(x_intervals), info.node_num, history_steps, event_num), dtype=np.float32
    )
    res_logs = []
    res = {}
    res["logs"] = []  # log sequences, array, shape is [num_chunks, ]
    res["stats"] = (
        res_stats  # log statistics, array, shape is [num_chunks, history_steps, node_num, event_num]
    )

    no_log_chunk = 0
    for chunk_idx, (s, e) in enumerate(x_intervals):
        s = float(s)
        e = float(e)
        if (chunk_idx + 1) % 100 == 0:
            print(
                "Transform chunk's logs into features of the {}/{}".format(
                    chunk_idx + 1, len(x_intervals)
                )
            )
        try:
            rows = df.loc[(df["Timestamp"] >= s) & (df["Timestamp"] < e)]
        except:
            no_log_chunk += 1
            continue
        # 'Service', 'Timestamp', 'Event', 'Content', 'EventTemplate'
        if log_process_type == "template":
            # Split logs by aligned time step.
            chunk_logs = []
            start_step = s
            for timestep in range(history_steps):
                # Select logs in the current time step.
                tmp_rows = rows.loc[
                    (rows["Timestamp"] >= start_step + timestep)
                    & (rows["Timestamp"] < start_step + timestep + 1)
                ]
                # Extract log templates used as discrete log events.
                logs = tmp_rows["EventTemplate"].tolist()
                chunk_logs.append(logs)
            # Append this chunk's log sequence.
            res_logs.append(chunk_logs)
            res_stats = None
        elif log_process_type == "template_statistics":
            chunk_logs = {}
            start_step = s
            for timestep in range(history_steps):
                chunk_logs[timestep] = {}
                tmp_rows = rows.loc[
                    (rows["Timestamp"] >= start_step + timestep)
                    & (rows["Timestamp"] < start_step + timestep + 1)
                ]
                service_events = tmp_rows.groupby("Service")
                for service, sgroup in service_events:
                    chunk_logs[timestep][info.service2id[service]] = sgroup[
                        "EventTemplate"
                    ].tolist()
                    events = sgroup.groupby("Event")
                    for event, egroup in events:
                        eid = 0 if event not in event2id else event2id[event]
                        res_stats[
                            chunk_idx, info.service2id[service], timestep, eid
                        ] += len(egroup)
            res_logs.append(chunk_logs)

        elif log_process_type == "raw":
            chunk_logs = {}
            start_step = s
            for timestep in range(history_steps):
                chunk_logs[timestep] = {}
                tmp_rows = rows.loc[
                    (rows["Timestamp"] >= start_step + timestep)
                    & (rows["Timestamp"] < start_step + timestep + 1)
                ]
                service_events = tmp_rows.groupby("Service")
                for service, sgroup in service_events:
                    chunk_logs[timestep][info.service2id[service]] = sgroup[
                        "EventTemplate"
                    ].tolist()

            res_logs.append(chunk_logs)
            res_stats = None
        elif log_process_type == "statistics":
            start_step = s
            for timestep in range(history_steps):
                tmp_rows = rows.loc[
                    (rows["Timestamp"] >= start_step + timestep)
                    & (rows["Timestamp"] < start_step + timestep + 1)
                ]
                service_events = tmp_rows.groupby("Service")
                for service, sgroup in service_events:
                    events = sgroup.groupby("Event")
                    for event, egroup in events:
                        eid = 0 if event not in event2id else event2id[event]
                        res_stats[
                            chunk_idx, info.service2id[service], timestep, eid
                        ] += len(egroup)

        else:
            pass
    if res_stats is not None:
        res_stats = np.transpose(res_stats, (0, 2, 1, 3))
    res["stats"] = res_stats
    res["logs"] = res_logs

    print("# Empty log:", no_log_chunk)
    with open(os.path.join(processed_path, cat, subdir + "-logs.pkl"), "wb") as fw:
        pickle.dump(res, fw)
    return res

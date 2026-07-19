"""
Dataset metadata utilities for the M3RL preprocessing pipeline.

`Dataset_Info` centralizes dataset-specific constants needed to transform raw
telemetry into model-ready tensors: service names, service-to-index maps,
static service dependencies, log parser settings, and paths to raw metric/log/
trace files. The model later treats services as graph nodes, so this stable
node ordering keeps metrics, logs, traces, labels, and adjacency matrices
aligned.
"""

import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import time
import json
from utils.utils import get_subfiles_subfolders, get_dirname
from logging import raiseExceptions


class Dataset_Info:
    """Container for raw dataset metadata and service graph definitions."""

    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        tmp_folder = os.getcwd()
        if tmp_folder.endswith("data"):
            datafolder = tmp_folder
        else:
            datafolder = os.path.join(tmp_folder, "data")
        if dataset_path == None:
            self.dataset_path = os.path.join(datafolder, dataset_name)
        else:
            self.dataset_path = dataset_path

        if self.dataset_name.find("Eadro") != -1:
            if self.dataset_name.find("TT") != -1:
                tmp_apiList = [
                    "assurance",
                    "auth",
                    "basic",
                    "cancel",
                    "config",
                    "contacts",
                    "food-map",
                    "food",
                    "inside-payment",
                    "notification",
                    "order-other",
                    "order",
                    "payment",
                    "preserve",
                    "price",
                    "route-plan",
                    "route",
                    "seat",
                    "security",
                    "station",
                    "ticketinfo",
                    "train",
                    "travel-plan",
                    "travel",
                    "travel2",
                    "user",
                    "verification-code",
                ]
                self.apiList = sorted(
                    ["ts-{}-service".format(api) for api in tmp_apiList]
                )
                edge_info = {
                    "preserve": [
                        "preserve",
                        "seat",
                        "security",
                        "food",
                        "order",
                        "ticketinfo",
                        "travel",
                        "contacts",
                        "notification",
                        "user",
                        "station",
                    ],
                    "seat": ["seat", "order", "config", "travel"],
                    "cancel": ["inside-payment", "order-other", "order"],
                    "security": ["security", "order-other", "order"],
                    "food": ["travel", "food-map", "station"],
                    "travel": ["travel", "order", "ticketinfo", "train", "route"],
                    "inside-payment": ["payment", "order"],
                    "ticketinfo": ["ticketinfo", "basic"],
                    "basic": ["basic", "route", "price", "train", "station"],
                    "order-other": ["station"],
                    "order": ["order", "station", "assurance"],
                    "auth": ["auth", "verification-code"],
                }
                self.edge_info = {
                    "ts-{}-service".format(k): ["ts-{}-service".format(vi) for vi in v]
                    for k, v in edge_info.items()
                }
            elif self.dataset_name.find("SN") != -1:
                self.apiList = sorted(
                    [
                        "social-graph-service",
                        "compose-post-service",
                        "post-storage-service",
                        "user-timeline-service",
                        "url-shorten-service",
                        "user-service",
                        "media-service",
                        "text-service",
                        "unique-id-service",
                        "user-mention-service",
                        "home-timeline-service",
                        "nginx-web-server",
                    ]
                )
                self.edge_info = {
                    "compose-post-service": [
                        "compose-post-service",
                        "home-timeline-service",
                        "media-service",
                        "post-storage-service",
                        "text-service",
                        "unique-id-service",
                        "user-service",
                        "user-timeline-service",
                    ],
                    "home-timeline-service": [
                        "home-timeline-service",
                        "post-storage-service",
                        "social-graph-service",
                    ],
                    "post-storage-service": ["post-storage-service"],
                    "social-graph-service": ["social-graph-service", "user-service"],
                    "text-service": [
                        "text-service",
                        "url-shorten-service",
                        "user-mention-service",
                    ],
                    "user-service": ["user-service"],
                    "user-timeline-service": ["user-timeline-service"],
                    "nginx-web-server": [
                        "compose-post-service",
                        "home-timeline-service",
                        "nginx-web-server",
                        "social-graph-service",
                        "user-service",
                    ],
                }
            self.log_template = "<Service> <Timestamp> <Content>"
            self.drain_similarity_threshold = 0.5  # Similarity threshold
            self.drain_depth = 4  # Depth of all leaf nodes
            self.metric_names = [
                "cpu_usage_system",
                "cpu_usage_total",
                "cpu_usage_user",
                "memory_usage",
                "memory_working_set",
                "rx_bytes",
                "tx_bytes",
            ]
            self.service_names = self.apiList
            self.service2id = {s: idx for idx, s in enumerate(self.service_names)}
            self.node_num = len(self.service_names)
            self.get_edges()
        elif self.dataset_name.find("Germany") != -1:
            self.apiList = ["wally113", "wally117", "wally122", "wally123", "wally124"]
            self.edge_info = {}
            self.log_template = "<Service> <Timestamp> <Level> <Content>"
            self.drain_similarity_threshold = 0.5
            self.drain_depth = 4
            self.metric_names = [
                "cpu.user",
                "mem.used",
                "load.cpucore",
                "load.min1",
                "load.min5",
                "load.min15",
            ]
            self.service_names = self.apiList
            self.service2id = {s: idx for idx, s in enumerate(self.service_names)}
            self.node_num = len(self.service_names)
            self.get_edges()
        else:
            raiseExceptions("Not Implemented yet {}".format(self.dataset_name))

        self.regex = [
            r'(?<=")[0-9a-zA-Z]+[-_]{1,}[0-9a-zA-Z-_]+(?=")',  # numbers-char-_
            r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)",  # IP
            r"(?<=[^A-Za-z0-9])(\-?\+?\d+\.*\d+)(?=[^A-Za-z0-9])|[0-9]+$",  # Numbers
            r"((?<=[^A-Za-z0-9])|^)(([0-9a-f]{2,}:){3,}([0-9a-f]{2,}))((?=[^A-Za-z0-9])|$)",  # ID
            r"((?<=[^A-Za-z0-9])|^)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})((?=[^A-Za-z0-9])|$)",  # IP
            r"((?<=[^A-Za-z0-9])|^)([0-9a-f]{6,} ?){3,}((?=[^A-Za-z0-9])|$)",  # SEQ
            r"((?<=[^A-Za-z0-9])|^)([0-9A-F]{4} ?){4,}((?=[^A-Za-z0-9])|$)",  # SEQ
            r"((?<=[^A-Za-z0-9])|^)(0x[a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)",  # HEX
            r"((?<=[^A-Za-z0-9])|^)([\-\+]?\d+)((?=[^A-Za-z0-9])|$)",  # NUM
            r'(?<=executed cmd )(".+?")',  # CMD
        ]

        self.metadata = {
            "node_num": self.node_num,
            "metric_num": len(self.metric_names),
        }

        files, dirs = get_subfiles_subfolders(self.dataset_path)
        self.insert_info("dataset_path", dataset_path)
        self.set_data_files_path(files, dirs)
        self.set_ts_metadata_info(files, dirs)

    def insert_info(self, key, value):
        self.metadata[key] = value

    def get_edges(self):
        src, des = [], []
        for s, v in self.edge_info.items():
            sid = self.service2id[s]
            for t in v:
                src.append(sid)
                des.append(self.service2id[t])
        self.edges = [src, des]

    def set_data_files_path(self, files, dirs):
        self.data_path = {}
        self.data_path["metrics"] = {}  # {SN.2022... : [metrics.file1, file2...],...}
        self.data_path["logs"] = {}
        self.data_path["traces"] = {}

        if self.dataset_name.find("Eadro") != -1:
            for dir in dirs:
                if dir.endswith("fault"):
                    # 'SN.2022...'
                    _, self.fault_data_subdirs = get_subfiles_subfolders(
                        os.path.join(self.dataset_path, "fault")
                    )
                elif dir.endswith("faultless"):
                    _, self.faultless_data_subdirs = get_subfiles_subfolders(
                        os.path.join(self.dataset_path, "faultless")
                    )
                else:
                    pass
            for dirs in [self.fault_data_subdirs, self.faultless_data_subdirs]:
                if dirs == self.fault_data_subdirs:
                    cat_name = "fault"
                else:
                    cat_name = "faultless"

                metric_data_path_list = []
                log_data_path_list = []
                trace_data_path_list = []
                for dir in dirs:
                    # [../logs.json,../spans.json], [../metrics]
                    log_trace_files, data_subdirs = get_subfiles_subfolders(dir)
                    # logs, traces
                    for file in log_trace_files:
                        file_name = os.path.basename(file)
                        if file_name == "logs.json":
                            log_data_path_list.append(file)
                        elif file_name == "spans.json":
                            trace_data_path_list.append(file)
                    # metrics directory
                    for sub_dir in data_subdirs:
                        metric_files, _ = get_subfiles_subfolders(sub_dir)
                        for file in metric_files:
                            metric_data_path_list.append(file)
                self.data_path["metrics"][cat_name] = metric_data_path_list
                self.data_path["logs"][cat_name] = log_data_path_list
                self.data_path["traces"][cat_name] = trace_data_path_list

        elif self.dataset_name.find("Germany") != -1:
            for dir in dirs:
                if os.path.basename(dir).startswith("concurrent"):
                    _, self.con_data_subdirs = get_subfiles_subfolders(
                        os.path.join(self.dataset_path, "concurrent data")
                    )
                else:
                    _, self.seq_data_subdirs = get_subfiles_subfolders(
                        os.path.join(self.dataset_path, "sequential_data")
                    )

            for dirs in [self.con_data_subdirs, self.seq_data_subdirs]:
                if dirs == self.con_data_subdirs:
                    cat_name = "con"
                else:
                    cat_name = "seq"
                metric_data_path_list = []
                log_data_path_list = []
                trace_data_path_list = []
                # /logs, metrics, traces
                for subdir in dirs:
                    if os.path.basename(subdir) == "metrics":
                        metrics_files, _ = get_subfiles_subfolders(subdir)
                        for file in metrics_files:
                            metric_data_path_list.append(file)
                    elif os.path.basename(subdir) == "logs":
                        log_files, _ = get_subfiles_subfolders(subdir)
                        for file in log_files:
                            if os.path.basename(file).startswith("logs"):
                                # logs_aggregated_concurrent.csv
                                log_data_path_list.append(file)
                    elif os.path.basename(subdir) == "traces":
                        _, trace_subdirs = get_subfiles_subfolders(subdir)
                        for trace_subdir in trace_subdirs:
                            trace_files, _ = get_subfiles_subfolders(trace_subdir)
                            for trace_file in trace_files:
                                trace_data_path_list.append(trace_file)

                self.data_path["metrics"][cat_name] = metric_data_path_list
                self.data_path["logs"][cat_name] = log_data_path_list
                self.data_path["traces"][cat_name] = trace_data_path_list

        self.insert_info("data_path", self.data_path)

    def set_ts_metadata_info(self, files, dirs):
        if self.dataset_name.find("Eadro") != -1:
            for dir in dirs:
                if dir.endswith("fault"):
                    self.fault_metadata_files, _ = get_subfiles_subfolders(
                        os.path.join(self.dataset_path, "fault")
                    )
                elif dir.endswith("faultless"):
                    self.faultless_metadata_files, _ = get_subfiles_subfolders(
                        os.path.join(self.dataset_path, "faultless")
                    )
                else:
                    pass
            for metadata_files in [
                self.fault_metadata_files,
                self.faultless_metadata_files,
            ]:
                if metadata_files == self.fault_metadata_files:
                    flag = "fault"
                else:
                    flag = "faultless"
                for file in metadata_files:
                    file_name, file_extension = os.path.splitext(file)
                    if (
                        file.endswith("json")
                        and not file.endswith("revised.json")
                        and not file.endswith("processed.json")
                    ):
                        fp = open(file, "rb")
                        data = json.load(fp)
                        # set time unit to second by int
                        result = {
                            "start": int(data["start"]),
                            "end": int(data["end"]),
                            "faults": [],
                        }
                        for item in data["faults"]:
                            service_name = "-".join(item["name"].split("-")[1:-1])
                            if self.dataset_name.find("SN") != -1:
                                if service_name == "nginx-thrift":
                                    service_name = "nginx-web-server"
                            else:
                                service_name = "ts-" + service_name + "-service"
                            result["faults"].append(
                                {
                                    "service": service_name,
                                    "s": int(item["start"]),
                                    "e": int((item["start"] + item["duration"])),
                                    "duration": item["duration"],
                                }
                            )
                        self.insert_info(flag + "_" + file_name, result)
                        out_file = file.split(".json")[0] + "_revised.json"
                        with open(out_file, "w") as out_fp:
                            json.dump(result, out_fp, indent=2)

        elif self.dataset_name.find("Germany") != -1:
            for dir in dirs:
                if dir.endswith("concurrent data"):
                    self.con_metadata_files, _ = get_subfiles_subfolders(
                        os.path.join(self.dataset_path, "concurrent data")
                    )
                else:
                    self.seq_metadata_files, _ = get_subfiles_subfolders(
                        os.path.join(self.dataset_path, "sequential_data")
                    )
            for metadata_files in [self.con_metadata_files, self.seq_metadata_files]:
                if metadata_files == self.con_metadata_files:
                    flag = "concurrent data"
                else:
                    flag = "sequential_data"
                for file in metadata_files:
                    file_name, file_extension = os.path.splitext(file)
                    if (
                        os.path.basename(file)
                        == "IMPORTANT_experiment_start_end_data.txt"
                    ):
                        with open(file, "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.strip().startswith("START OF EXPERIMENTS"):
                                    trace_start = line.strip().split("-", 1)[1].strip()
                                    trace_start = time.mktime(
                                        time.strptime(trace_start, "%Y-%m-%d %H:%M:%S")
                                    )
                                elif line.strip().startswith("END OF EXPERIMENTS"):
                                    trace_end = line.strip().split("-", 1)[1].strip()
                                    trace_end = time.mktime(
                                        time.strptime(trace_end, "%Y-%m-%d %H:%M:%S")
                                    )
                                elif line.strip().startswith("START") and (
                                    not line.strip().startswith("START OF EXPERIMENTS")
                                ):
                                    log_metric_start = (
                                        line.strip().split(":", 1)[1].strip()
                                    )
                                    log_metric_start = time.mktime(
                                        time.strptime(
                                            log_metric_start, "%Y-%m-%d %H:%M:%S"
                                        )
                                    )
                                elif line.strip().startswith("END") and (
                                    not line.strip().startswith("END OF EXPERIMENTS")
                                ):
                                    log_metric_end = (
                                        line.strip().split(":", 1)[1].strip()
                                    )
                                    log_metric_end = time.mktime(
                                        time.strptime(
                                            log_metric_end, "%Y-%m-%d %H:%M:%S"
                                        )
                                    )
                            # set time unit to second by int
                            result = {
                                "trace_start": trace_start,
                                "trace_end": trace_end,
                                "log_metric_start": log_metric_start,
                                "log_metric_end": log_metric_end,
                            }
                            self.insert_info(
                                flag + "_" + os.path.basename(file).split(".")[0],
                                result,
                            )
                            out_file = file.split(".txt")[0] + "_revised.json"
                            with open(out_file, "w") as out_fp:
                                json.dump(result, out_fp, indent=2)

        else:
            pass


def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith(".json")
        with open(filepath, "r") as f:
            content = f.read()
            return json.loads(content)
    else:
        raiseExceptions("File path " + filepath + " not exists!")
        return


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


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

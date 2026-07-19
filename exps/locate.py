"""
Command-line entry point for the Stage 2 root cause localization task.

The script loads Stage 1 pretrained modules into `M3RLStage2`, fine-tunes the
localization head, and evaluates service-level root cause rankings/logits.
"""

import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))

if sys.platform.startswith("win"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import math
import time
import wandb
import argparse
import tqdm
import configparser
from metrics.metrics import Metrics
from datetime import datetime
from torch.utils.data import DataLoader
from engines.locate_engine import trainer, load_pretrained_weights
from utils.tools import EarlyStopping, SettingsFromArgs, SettingsFromConfigs
from utils.utils import *
import numpy as np
from packaging import version

np_version = np.__version__
if version.parse(np_version) < version.parse("2.0"):
    INF = np.Inf
else:
    INF = np.inf

parser = argparse.ArgumentParser(description="arguments")
parser.add_argument("--no_cuda", action="store_true", help="NO GPU")
parser.add_argument("--task", type=str, default="locate", help="Task name")
parser.add_argument(
    "--model", type=str, default="M3RLStage2", help="The wanted model to run"
)
parser.add_argument(
    "--dataset", type=str, default="SN-Eadro Dataset", help="Dataset config directory"
)
parser.add_argument("--datapath", type=str, default="../data", help="Data dir")
parser.add_argument(
    "--sample_rate", type=int, default=5, help="The sample rate of collecting metrics"
)
parser.add_argument("--batch_size", type=int, default=10, help="Training Batch Size")
parser.add_argument(
    "--valid_batch_size", type=int, default=10, help="Validation batch size"
)
parser.add_argument("--test_batch_size", type=int, default=10, help="Test Batch Size")
parser.add_argument(
    "--if_shuffle", type=eval, default=True, help="Whether to shuffle dataset"
)
parser.add_argument(
    "--if_scale",
    type=eval,
    default=False,
    help="Whether to scale dataset before dataloader",
)
parser.add_argument(
    "--if_test",
    type=eval,
    default=False,
    help="Loading small dataset when test the script",
)
parser.add_argument(
    "--num_workers", type=int, default=8, help="The num_workers of dataloader"
)

args = parser.parse_args()
args_dict = vars(args)
settings = SettingsFromArgs(args_dict)

print("torch.cuda.is_available(): ", torch.cuda.is_available())
device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device1)


def locate(data_path, model_name, dataset_name, log, device, settings):
    root_path = data_path.replace(os.path.basename(data_path), "")
    print(root_path)
    print("Loading {} Dataset...".format(dataset_name))
    dataset = prepare_and_load_dataset(
        settings=settings,
        device=device,
        dataset_name=dataset_name,
        data_path=data_path,
        sample_rate=settings.sample_rate,
        if_scale=settings.if_scale,
        if_test=settings.if_test,
    )
    task = settings.task
    if settings.if_scale:
        scaler = dataset["scaler"]
    else:
        scaler = None

    log_string(
        log, "Experiment Date: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    log_string(log, "Loading Data ...")
    log_string(log, f'x_train: {len(dataset["train"])}\t\t')
    log_string(log, f'x_val:   {len(dataset["val"])}\t\t')
    log_string(log, f'x_test:   {len(dataset["test"])}\t\t')
    log_string(log, "Data Loaded !!")

    engine = trainer(settings, log, device)
    log_string(log, "Beginning training model ...")

    his_loss = []
    val_time = []
    train_time = []

    wait = 0
    val_mae_min = float("inf")
    best_model_wts = None
    valid_epoch_interval = settings.valid_epoch_interval

    if sys.platform.startswith("win"):
        num_workers = 0
    else:
        num_workers = settings.num_workers

    metrics_train = Metrics(
        settings=settings,
        cat="train",
        dataset_name=dataset_name,
        task=task,
        scaler=scaler,
        normalize_type=settings.normalize_type,
    )
    metrics_valid = Metrics(
        settings=settings,
        cat="val",
        dataset_name=dataset_name,
        task=task,
        scaler=scaler,
        normalize_type=settings.normalize_type,
    )
    metrics_test = Metrics(
        settings=settings,
        cat="test",
        task=task,
        dataset_name=dataset_name,
        scaler=scaler,
        normalize_type=settings.normalize_type,
    )

    dataloader_train = DataLoader(
        dataset["train"],
        batch_size=settings.batch_size,
        shuffle=settings.if_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=settings.drop_last,
    )
    dataloader_val = DataLoader(
        dataset["val"],
        batch_size=settings.valid_batch_size,
        shuffle=settings.if_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=settings.drop_last,
    )
    dataloader_test = DataLoader(
        dataset["test"],
        batch_size=settings.test_batch_size,
        shuffle=settings.if_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=settings.drop_last,
    )

    if settings.if_wandb:
        wandb.init(
            config={
                "architecture": "diffusion",
                "epoch": settings.epochs,
            },
            project="M3RL",
            entity="xyf123",
            name="{}-{}".format(model_name, settings.expid),
            notes=dataset_name,
            job_type="training",
            reinit=True,
        )

    early_stopping = EarlyStopping(
        task=task,
        expid=settings.expid,
        patience=settings.patience,
        verbose=True,
        save_limit=settings.save_limit,
    )
    loss_epoch_list = []
    nan_count = 0
    inf_count = 0

    if model_name == "M3RLStage2":
        pretrain1_model_name = "M3RLStage1"
        save_path1 = os.path.join(
            root_path, "checkpoints", dataset_name, pretrain1_model_name
        )
        model_path1 = early_stopping.find_best_model(save_path1)
        if model_path1 is None:
            sys.exit(556)
        load_pretrained_weights(
            engine.model, stage=2, pretrained_path=model_path1, device=device
        )

    for epoch in tqdm.tqdm(
        range(1, settings.epochs + 1), desc="Epoch Bar", dynamic_ncols=True
    ):
        metrics_train.clear_list()

        t1 = time.time()
        iters_to_accumulate = settings.iters_to_accumulate
        len_train = len(dataloader_train)  # num_samples/batch_size
        lr_decay = settings.lr_decay

        for ix, data in tqdm.tqdm(
            enumerate(dataloader_train), desc="Iter Bar", dynamic_ncols=True
        ):
            loss_epoch = 0
            if dataset_name.find("TT-Ebpf") != -1:
                x, y, adj_x, adj_y, l2, l3, instances_index_vm, l1 = data
                x_train = torch.Tensor(x).to(device)  # B T N C
                y_train = torch.Tensor(y).to(device)
                adj_x = torch.Tensor(adj_x).to(device)  # B T N N
                adj_y = torch.Tensor(adj_y).to(device)
                labels = torch.Tensor(l2).to(device)
                instances_index_vm = torch.Tensor(instances_index_vm).to(device)
            else:
                x_m = torch.Tensor(data["metrics"]).to(device)
                x_l = torch.Tensor(data["logs"]).to(device)
                x_t = torch.Tensor(data["traces"]).to(device)
                x_train = (x_m, x_l, x_t)
                y_train = torch.Tensor(data["y_labels"]).to(device)
                labels = torch.Tensor(data["labels"]).to(device)
                if settings.if_dynamic_adj:
                    adj_x = torch.Tensor(data["adj"]).to(device)
                else:
                    adj_x = torch.Tensor(data["static_adj"]).to(device)
                adj_y = None
                instances_index_vm = None

            metrics_train, loss_epoch = engine.train_model(
                x=x_train,
                y=y_train,
                x_mark_enc=None,
                x_mark_dec=None,
                labels=labels,
                adj_x=adj_x,
                adj_y=adj_y,
                instances_index_vm=instances_index_vm,
                ix=ix,
                iters_to_accumulate=iters_to_accumulate,
                len_train=len_train,
                loss_epoch=loss_epoch,
                learning_rate=settings.learning_rate,
                epoch=epoch,
                metrics=metrics_train,
            )
            if math.isnan(metrics_train.metrics["loss"]):
                nan_count += 1
                logs = "nan occurs {:03d} times."
                log_string(log, logs.format(nan_count))
            elif metrics_train.metrics["loss"] == float("inf"):
                inf_count += 1
                logs = "inf occurs {:03d} times."
                log_string(log, logs.format(inf_count))
            else:
                loss_epoch_list.append(loss_epoch)
                metrics_train.update_metrics_list()
                if ix % settings.print_every == 0:
                    logs = "Iter: {:03d}, Train Loss: {:.4f}, lr: {}.\n"
                    log_string(
                        log,
                        logs.format(
                            ix,
                            metrics_train.loss_list[-1],
                            engine.optimizer.param_groups[0]["lr"],
                        ),
                    )
                    print("nan occurs {:03d} times.".format(nan_count))
                    print("inf occurs {:03d} times.".format(inf_count))
                    if settings.if_wandb:
                        wandb.log(
                            {
                                "Iter": ix,
                                "Train Loss": metrics_train.loss_list[-1],
                                "lr": engine.optimizer.param_groups[0]["lr"],
                            }
                        )

        if lr_decay:
            engine.lr_scheduler.step()
            print(engine.lr_scheduler.get_last_lr())

        t2 = time.time()
        train_time.append(t2 - t1)
        logs = "\nEpoch: {:03d}, Epoch Train Loss: {:.4f}"
        print(logs.format(epoch, loss_epoch_list[epoch - 1] / len_train), flush=True)
        if settings.if_wandb:
            wandb.log(
                {
                    "Epoch": epoch,
                    "Epoch Train Loss": loss_epoch_list[epoch - 1] / len_train,
                }
            )

        if epoch % valid_epoch_interval == 0:
            metrics_valid.clear_list()
            s1 = time.time()
            print("Valid phrase of epoch {} begin...".format(epoch))
            for ix, data in tqdm.tqdm(
                enumerate(dataloader_val), desc="Iter Bar", dynamic_ncols=True
            ):
                if dataset_name.find("TT-Ebpf") != -1:
                    x, y, adj_x, adj_y, l2, l3, instances_index_vm, l1 = data
                    x_val = torch.Tensor(x).to(device)  # B T N C
                    y_val = torch.Tensor(y).to(device)
                    adj_x = torch.Tensor(adj_x).to(device)  # B T N N
                    adj_y = torch.Tensor(adj_y).to(device)
                    labels = torch.Tensor(l2).to(device)
                    instances_index_vm = torch.Tensor(instances_index_vm).to(device)
                else:
                    x_m = torch.Tensor(data["metrics"]).to(device)
                    x_l = torch.Tensor(data["logs"]).to(device)
                    x_t = torch.Tensor(data["traces"]).to(device)
                    x_val = (x_m, x_l, x_t)
                    y_val = torch.Tensor(data["y_labels"]).to(device)
                    labels = torch.Tensor(data["labels"]).to(device)
                    if settings.if_dynamic_adj:
                        adj_x = torch.Tensor(data["adj"]).to(device)
                    else:
                        adj_x = torch.Tensor(data["static_adj"]).to(device)
                    adj_y = None
                    instances_index_vm = None

                metrics_valid = engine.eval_model(
                    x=x_val,
                    y=y_val,
                    x_mark_enc=None,
                    x_mark_dec=None,
                    labels=labels,
                    adj_x=adj_x,
                    adj_y=adj_y,
                    instances_index_vm=instances_index_vm,
                    metrics=metrics_valid,
                    epoch=epoch,
                )

                metrics_valid.update_metrics_list()

                logs = "Epoch: {:03d}, Valid Loss: {:.4f}.\n"
                log_string(log, logs.format(epoch, metrics_valid.loss_list[-1]))
                if settings.if_wandb:
                    wandb.log(
                        {
                            "Valid Epoch": epoch,
                            "Valid Loss": metrics_valid.loss_list[-1],
                        }
                    )

            s2 = time.time()
            logs = "Epoch: {:03d}, Inference Time: {:.4f} secs"
            log_string(log, logs.format(epoch, (s2 - s1)))

            val_time.append(s2 - s1)
            metrics_train.calc_mean_for_list()
            metrics_valid.calc_mean_for_list()
            metrics_train.calc_precision_rank_metric()
            metrics_valid.calc_precision_rank_metric()
            his_loss.append(metrics_valid.loss_list[-1])

            logs = (
                "\nEpoch: {:03d}, Averge metrics:\n "
                "Train Loss: {:.4f}, Train F1: {:.4f}, Train REC: {:.4f}, \n"
                "Train PRE: {:.4f}, Train HR@1: {:.4f}, Train HR@3: {:.4f}, Train HR@5: {:.4f}\n"
                "Train NDCG@1: {:.4f}, Train NDCG@3: {:.4f}, Train NDCG@5: {:.4f}\n"
                "Train AvgHR@3: {:.4f}, Train AvgHR@3 Scores: {:.4f}\n"
                "Valid Loss: {:.4f}, Valid F1: {:.4f}, Valid REC: {:.4f}, \n"
                "Valid PRE: {:.4f}, Valid HR@1: {:.4f}, Valid HR@3: {:.4f}, Valid HR@5: {:.4f}\n"
                "Valid NDCG@1: {:.4f}, Valid NDCG@3: {:.4f}, Valid NDCG@5: {:.4f}\n"
                "Valid AvgHR@3: {:.4f}, Valid AvgHR@3 Scores: {:.4f}\n"
                "Training Time: {:.4f}/epoch"
            )
            log_string(
                log,
                logs.format(
                    epoch,
                    metrics_train.loss_list[-1],
                    metrics_train.f1_list[-1],
                    metrics_train.rec_list[-1],
                    metrics_train.pre_list[-1],
                    metrics_train.hr_1_list[-1],
                    metrics_train.hr_3_list[-1],
                    metrics_train.hr_5_list[-1],
                    metrics_train.ndcg_1_list[-1],
                    metrics_train.ndcg_3_list[-1],
                    metrics_train.ndcg_5_list[-1],
                    metrics_train.avg_hr_3_list[-1],
                    metrics_train.avg_hr_3_scores_list[-1],
                    metrics_valid.loss_list[-1],
                    metrics_valid.f1_list[-1],
                    metrics_valid.rec_list[-1],
                    metrics_valid.pre_list[-1],
                    metrics_valid.hr_1_list[-1],
                    metrics_valid.hr_3_list[-1],
                    metrics_valid.hr_5_list[-1],
                    metrics_valid.ndcg_1_list[-1],
                    metrics_valid.ndcg_3_list[-1],
                    metrics_valid.ndcg_5_list[-1],
                    metrics_valid.avg_hr_3_list[-1],
                    metrics_valid.avg_hr_3_scores_list[-1],
                    t2 - t1,
                ),
            )

            if dataset_name in ["TT-Ebpf Dataset", "SN-Ebpf Dataset"]:
                save_path = os.path.join(
                    root_path,
                    "checkpoints",
                    dataset_name,
                    str(settings.sample_rate),
                    model_name,
                )
                save_loss_path = os.path.join(
                    root_path,
                    "losses",
                    dataset_name,
                    str(settings.sample_rate),
                    model_name,
                )
            else:
                save_path = os.path.join(
                    root_path, "checkpoints", dataset_name, model_name
                )
                save_loss_path = os.path.join(
                    root_path, "losses", dataset_name, model_name
                )
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(save_loss_path):
                os.makedirs(save_loss_path)

            early_stopping(metrics_valid.f1_list[-1], engine.model, save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            save_loss_np_path = os.path.join(
                save_loss_path, settings.model + "_history_loss" + f"_{settings.expid}"
            )
            np.save(save_loss_np_path, his_loss)

    log_string(log, "Training Completed ...")
    log_string(
        log,
        "The Validation F1 of the best model is "
        + str(round(early_stopping.val_loss_min, 3)),
    )
    log_string(
        log, "Average Epoch Training Loss: {:.4f}".format(np.mean(loss_epoch_list))
    )
    log_string(
        log, "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time))
    )
    log_string(
        log, "Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time))
    )

    log_string(log, "Testing Model ...")

    if dataset_name in ["trainticket", "socketshop"]:
        save_path = os.path.join(
            root_path,
            "checkpoints",
            dataset_name,
            str(settings.sample_rate),
            model_name,
        )
    else:
        save_path = os.path.join(root_path, "checkpoints", dataset_name, model_name)

    if early_stopping.val_loss_min == INF:
        model_path = early_stopping.find_best_model(save_path)
        if model_path is None:
            sys.exit(555)
    else:
        print_val_loss_min = "{:.3f}".format(early_stopping.val_loss_min)
        model_file = (
            "exp_" + str(settings.expid) + "_" + print_val_loss_min + "_best_model.pth"
        )
        model_path = os.path.join(save_path, model_file)
    engine.model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location="cuda:0")
    )

    s1 = time.time()
    samples, targets = [], []
    sv_list = []
    for ix, data in tqdm.tqdm(
        enumerate(dataloader_test), desc="Iter Bar", dynamic_ncols=True
    ):
        if dataset_name.find("TT-Ebpf") != -1:
            x, y, adj_x, adj_y, l2, l3, instances_index_vm, l1 = data
            x_test = torch.Tensor(x).to(device)  # B T N C
            y_test = torch.Tensor(y).to(device)
            adj_x = torch.Tensor(adj_x).to(device)  # B T N N
            adj_y = torch.Tensor(adj_y).to(device)
            labels = torch.Tensor(l2).to(device)
            instances_index_vm = torch.Tensor(instances_index_vm).to(device)
        else:
            x_m = torch.Tensor(data["metrics"]).to(device)
            x_l = torch.Tensor(data["logs"]).to(device)
            x_t = torch.Tensor(data["traces"]).to(device)
            x_test = (x_m, x_l, x_t)
            y_test = torch.Tensor(data["y_labels"]).to(device)
            labels = torch.Tensor(data["labels"]).to(device)
            if settings.if_dynamic_adj:
                adj_x = torch.Tensor(data["adj"]).to(device)
            else:
                adj_x = torch.Tensor(data["static_adj"]).to(device)
            adj_y = None
            instances_index_vm = None

        metrics_test, sv = engine.test_model(
            x=x_test,
            y=y_test,
            x_mark_enc=None,
            x_mark_dec=None,
            labels=labels,
            adj_x=adj_x,
            adj_y=adj_y,
            instances_index_vm=instances_index_vm,
            metrics=metrics_test,
            epoch=epoch,
            samples=samples,
            targets=targets,
        )
        sv_list.append(sv)
        metrics_test.update_metrics_list()

    pkl_name = (
        "exp_"
        + str(settings.expid)
        + "_"
        + task
        + "_"
        + str(round(early_stopping.val_loss_min, 3))
        + "_{}.pkl".format(model_name)
    )
    save_path = os.path.join(root_path, "saves", dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if model_name.find("M3RL") != -1:
        if_save_sv = settings.if_save_sv
        if if_save_sv:
            file_save_path = os.path.join(save_path, pkl_name)
            save_sv_result(sv_list, file_save_path)
    metrics_test.calc_mean_for_list()
    metrics_test.calc_precision_rank_metric()
    s2 = time.time()
    logs = "Test Inference Time: {:.4f} secs"
    log_string(log, logs.format((s2 - s1)))

    len_test = len(dataloader_test)  # num_samples/batch_size
    logs = "Average Test Inference Time: {:.4f} secs"
    avag_inference_time = (s2 - s1) / float(len_test)
    log_string(log, logs.format(avag_inference_time))

    logs = (
        "On average over {} horizons, Test F1: {:.4f}, Test REC: {:.4f}, Test PRE: {:.4f}, Test HR@1: {:.4f},\n"
        "Test HR@3: {:.4f}, Test HR@5: {:.4f}, Test NDCG@1: {:.4f}, Test NDCG@3: {:.4f},\n"
        "TEst NDCG@5: {:.4f}, Test AVGHR@3: {:.4f},  Test  AVGHR@3 Scores: {:.4f}.\n"
    )

    log_string(
        log,
        logs.format(
            settings.history_steps,
            metrics_test.f1_list[-1],
            metrics_test.rec_list[-1],
            metrics_test.pre_list[-1],
            metrics_test.hr_1_list[-1],
            metrics_test.hr_3_list[-1],
            metrics_test.hr_5_list[-1],
            metrics_test.ndcg_1_list[-1],
            metrics_test.ndcg_3_list[-1],
            metrics_test.ndcg_5_list[-1],
            metrics_test.avg_hr_3_list[-1],
            metrics_test.avg_hr_3_scores[-1],
        ),
    )
    print("The best model is {}".format(os.path.basename(model_path)))
    if settings.if_wandb:
        wandb.log(
            {
                "Test F1": metrics_test.f1_list[-1],
                "TEST REC": metrics_test.rec_list[-1],
                "Test PRE": metrics_test.pre_list[-1],
                "Test HR@1": metrics_test.hr_1_list[-1],
                "Test HR@3": metrics_test.hr_3_list[-1],
                "Test HR@5": metrics_test.hr_5_list[-1],
                "Test NDCG@1": metrics_test.ndcg_1_list[-1],
                "Test NDCG@3": metrics_test.ndcg_3_list[-1],
                "Test NDCG@5": metrics_test.ndcg_5_list[-1],
                "Test AVG HR@3": metrics_test.avg_hr_3_list[-1],
                "Test AVG HR@3 Scores": metrics_test.avg_hr_3_scores[-1],
                "Avag Inference Time": avag_inference_time,
            }
        )
        wandb.finish()
    log_string(
        log, "Testing of {} over {} is completed.".format(model_name, dataset_name)
    )


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_path, "data")
    config_path = os.path.join(root_path, "configs")
    config_file = os.path.join(config_path, args.dataset, f"{args.model}.conf")
    configs = configparser.ConfigParser()
    configs.read(config_file)
    settings = SettingsFromConfigs(configs)

    if args.dataset == "trainticket" or args.dataset == "socketshop":
        log_path = os.path.join(
            root_path,
            "logs",
            args.dataset,
            str(args.sample_rate),
            args.model,
            args.task,
        )
    else:
        log_path = os.path.join(root_path, "logs", args.dataset, args.model, args.task)
    if os.path.exists(log_path):
        pass
    else:
        os.makedirs(log_path)
    log_file_path = os.path.join(log_path, "train.log")
    log1 = open(log_file_path, "w")
    log_string(log1, str(args))
    start = time.time()
    locate(data_path, settings.model, settings.dataset, log1, device1, settings)
    end = time.time()

    log_string(log1, "total time: %.2fhours" % ((end - start) / 3600))
    log1.close()

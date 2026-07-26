import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))

import torch
import numpy as np
import time
from sklearn.metrics import ndcg_score


def MAE(true, pred):
    return np.mean(np.abs(pred - true))


def MSE(true, pred):
    return np.mean((pred - true) ** 2)


def RMSE(true, pred):
    return np.sqrt(MSE(true, pred))


def RMSLE(true, pred):
    # Masked RMSLE Loss
    pred = np.log(np.abs(pred) + 1)
    true = np.log(np.abs(true) + 1)
    return np.sqrt(MSE(true, pred))


def MAPE(true, pred):
    return np.mean(np.abs((pred - true) / (true + 1e-10)))


def MSPE(true, pred):
    return np.mean(np.square((pred - true) / (true + 1e-10)))


def mask_np(array, null_val=0):
    if np.isnan(null_val):
        return (~np.isnan(array)).astype("float32")
    else:
        return np.not_equal(array, null_val).astype("float32")


def masked_mae_np(y_true, y_pred, null_val=0):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))


def masked_mse_np(y_true, y_pred, null_val=0):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_rmse_np(y_true, y_pred, null_val=0):
    # Masked RMSE Loss
    return np.sqrt(masked_mse_np(y_true=y_true, y_pred=y_pred, null_val=null_val))


def masked_rmsle_np(y_true, y_pred, null_val=0):
    # Masked RMSLE Loss
    y_pred = np.log(np.abs(y_pred) + 1)
    y_true = np.log(np.abs(y_true) + 1)
    return np.sqrt(masked_mse_np(y_true=y_true, y_pred=y_pred, null_val=null_val))


def masked_mape_np(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / (y_true + 1e-10))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def masked_mspe_np(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.square((y_pred - y_true) / (y_true + 1e-10))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def metric(y_true, y_pred):
    mae = masked_mae_np(y_true, y_pred, 0).item()
    mse = masked_mse_np(y_true, y_pred, 0).item()
    rmse = masked_rmse_np(y_true, y_pred, 0).item()
    rmsle = masked_rmsle_np(y_true, y_pred, 0).item()
    mape = masked_mape_np(y_true, y_pred, 0).item()
    mspe = masked_mspe_np(y_true, y_pred, 0).item()

    return mae, mse, rmse, rmsle, mape, mspe


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_crps(target, forecast, eval_points):
    """
    target: (B, T, V), torch.Tensor
    forecast: (B, n_sample, T, V), torch.Tensor
    eval_points: (B, T, V): which values should be evaluated,
    """

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def mis(
    target: np.ndarray,
    lower_quantile: np.ndarray,
    upper_quantile: np.ndarray,
    alpha: float,
) -> float:
    r"""
    mean interval score
    Implementation comes form glounts.evalution metrics
    .. math::
    msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U])
    """
    numerator = np.mean(
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - target) * (target < lower_quantile)
        + 2.0 / alpha * (target - upper_quantile) * (target > upper_quantile)
    )
    return numerator


def calc_mis(target, forecast, alpha=0.05):
    """
    target: (B, T, V),
    forecast: (B, n_sample, T, V)
    """
    return mis(
        target=target.cpu().numpy(),
        lower_quantile=torch.quantile(forecast, alpha / 2, dim=1).cpu().numpy(),
        upper_quantile=torch.quantile(forecast, 1.0 - alpha / 2, dim=1).cpu().numpy(),
        alpha=alpha,
    )


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Metrics(object):
    """Computes and stores the average and current value; call reset before reuse."""

    def __init__(
        self, settings, cat, dataset_name, task, scaler=None, normalize_type=0
    ):
        self.time_start = time.time()
        self.task = task
        self.best_metrics = {
            "mae": np.inf,
            "mse": np.inf,
            "rmse": np.inf,
            "rmsle": np.inf,
            "mape": np.inf,
            "mspe": np.inf,
            "crps": np.inf,
            "mis": np.inf,
            "f1": np.inf,
            "rec": np.inf,
            "pre": np.inf,
            "hr_1": np.inf,
            "ndcg_1": np.inf,
            "hr_3": np.inf,
            "ndcg_3": np.inf,
            "hr_5": np.inf,
            "ndcg_5": np.inf,
            "avg_hr_3": np.inf,
            "avg_hr_3_scores": np.inf,
            "epoch": np.inf,
        }
        self.metrics = {}
        self.mean_metrics = {}
        self.loss_list = []
        self.mae_list = []
        self.mse_list = []
        self.rmse_list = []
        self.rmsle_list = []
        self.mape_list = []
        self.mspe_list = []
        self.crps_list = []
        self.mis_list = []
        self.f1_list = []
        self.rec_list = []
        self.pre_list = []
        self.hr_1_list = []
        self.ndcg_1_list = []
        self.hr_3_list = []
        self.ndcg_3_list = []
        self.hr_5_list = []
        self.ndcg_5_list = []
        self.avg_hr_3_list = []
        self.avg_hr_3_scores_list = []

        self.dataset_name = dataset_name
        self.scaler = scaler
        self.normalize_type = normalize_type

        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.TP = 0

        self.hrs, self.ndcgs = np.zeros(5, dtype=np.float32), np.zeros(
            5, dtype=np.float32
        )
        if cat == "train":
            self.avg_hr_3_scores = np.zeros(settings.batch_size, dtype=np.float32)
        elif cat == "val":
            self.avg_hr_3_scores = np.zeros(settings.valid_batch_size, dtype=np.float32)
        else:
            self.avg_hr_3_scores = np.zeros(settings.test_batch_size, dtype=np.float32)

    def update_metrics(self, y_true, y_pred, loss):
        """
        both y_true and y_pred should be numpy
        :param y_true: tensor or dict, (B, T_p, V, D)
        :param y_pred: tensor or dict, (B, n_samples, T_p, V, D) or (B, T_p, V, D) in tensor
        keys: locate_logits, detect_logits; values: (B, N), (B, 2) tensor
        :return:
        """
        if isinstance(y_pred, dict):
            if y_true.dim() == 1:
                B = y_true.shape  # B
            else:
                B, N = y_true.shape
            labels = y_true.detach().cpu().numpy().astype(int)  # B,
            self.metrics = {
                "f1": 0.0,
                "rec": 0.0,
                "pre": 0.0,
                "hr_1": 0.0,
                "ndcg_1": 0.0,
                "hr_3": 0.0,
                "ndcg_3": 0,
                "hr_5": 0.0,
                "ndcg_5": 0.0,
                "avg_hr_3": 0.0,
                "avg_hr_3_scores": 0.0,
                "loss": 0.0,
                "time": 0.0,
            }
            # TP, FP, FN = 0, 0, 0
            if (y_pred["locate_logits"] is not None) and (
                y_pred["detect_logits"] is not None
            ):
                locate_logits = y_pred["locate_logits"].detach().cpu().numpy()  # B N
                detect_logits = y_pred["detect_logits"].detach().cpu().numpy()  # B 2
                self.task = "locate"
                B, N = y_pred["locate_logits"].shape

                y_locate_prob = np.zeros((B, N))
                if len(labels.shape) == 1:
                    for i in range(B):
                        if labels[i] > -1:
                            y_locate_prob[i, labels[i]] = 1
                else:
                    y_locate_prob = labels.copy()

                y_detect = np.zeros((B,))
                if len(labels.shape) == 1:
                    #     y_detect[i] = int(labels[i] > -1)
                    y_detect = (labels != -1).astype(int)
                else:
                    y_detect = np.any(labels == 1, axis=1).astype(int)

                node_probs = softmax(locate_logits)  # B N
                node_list = np.flip(
                    node_probs.argsort(axis=1), axis=1
                )  # B N, node idx from high prob to low prob
                node_pred = []
                detect_pred = detect_logits.argmax(axis=1) # B
                for i in range(B):
                    node_pred.append([-1] if detect_pred[i] < 1 else node_list[i])

                for i, nodes_candidate in enumerate(node_pred):
                    if len(labels.shape) == 1:
                        true_anomaly = labels[i] != -1
                        culprit_nodes = [labels[i]] if true_anomaly else []
                    else:
                        tmp = np.where(labels[i] == 1)
                        if tmp[0].shape != (0,):
                            culprit_nodes = np.where(labels[i] == 1).tolist()
                            true_anomaly = len(culprit_nodes) > 0
                        else:
                            true_anomaly = False
                            culprit_nodes = []
                    if not true_anomaly:
                        # Normal label predicted as abnormal.
                        if nodes_candidate[0] != -1:
                            self.FP += 1
                        # Normal label predicted as normal.
                        else:
                            self.TN += 1
                    else:
                        # Abnormal label predicted as normal.
                        if nodes_candidate[0] == -1:
                            self.FN += 1
                        else:
                            # Abnormal label predicted as abnormal.
                            self.TP += 1
                            # Multi-label root-cause ranking case.
                            rank = None
                            for idx, node in enumerate(nodes_candidate):
                                if node in culprit_nodes:
                                    rank = idx
                                    break
                            if rank is not None:
                                for j in range(5):
                                    self.hrs[j] += int(rank <= j)
                                    self.ndcgs[j] += ndcg_score(
                                        [y_locate_prob[i]], [node_probs[i]], k=j + 1
                                    )
                                self.avg_hr_3_scores[i] = (
                                    1 / (rank + 1) if rank < 3 else 0
                                )
                self.avg_hr_3_scores_list.append(np.mean(self.avg_hr_3_scores))

            elif (y_pred["locate_logits"] is None) and (
                y_pred["detect_logits"] is not None
            ):
                detect_logits = y_pred["detect_logits"].detach().cpu().numpy()  # B 2
                self.task = "detect"
                y_detect = np.zeros((B,))
                if len(labels.shape) == 1:
                    for i in range(B):
                        # y_detect[i] = int(labels[i] > -1)
                        y_detect = (labels != -1).astype(int)
                else:
                    y_detect = np.any(labels == 1, axis=1).astype(int)

                detect_pred = detect_logits.argmax(axis=1).squeeze()  # B 2 -> B
                for i in range(B):
                    if y_detect[i] == 1:  # True anomaly
                        if detect_pred[i] == 1:  # Correctly detected
                            self.TP += 1
                        else:  # Missed detection
                            self.FN += 1
                    else:  # Normal
                        if detect_pred[i] == 1:  # False alarm
                            self.FP += 1

                self.hrs = None
                self.ndcgs = None
            else:
                # TP, FP, FN = 0, 0, 0
                self.hrs = None
                self.ndcgs = None

            #     'ndcg_5'] = self.calc_precision_rank_metric(self.TP, self.FP, self.FN, hrs, ndcgs)

        elif isinstance(y_pred, np.ndarray):
            self.task = "forecast"
            assert isinstance(y_true, np.ndarray) and isinstance(
                y_pred, np.ndarray
            ), f"y_true and y_pred should be np.ndarray, now its type is y_true:{type(y_true)}, y_pred:{type(y_pred)}"
            self.metrics = {
                "mae": 0.0,
                "mse": 0.0,
                "rmse": 0.0,
                "rmsle": 0.0,
                "mape": 0.0,
                "mspe": 0.0,
                "crps": 0,
                "mis": 0.0,
                "loss": 0.0,
                "time": 0.0,
            }

            if (
                y_pred.shape == y_true.shape
            ):  # y_true: (B, T_p, V, D) and y_pred: (B, T_p, V, D)
                y_pred = np.expand_dims(y_pred, axis=1)  # (B, 1, T_p, V, D)

            # probabilistic metric
            eval_points = np.ones_like(y_true)
            self.metrics["crps"] = calc_quantile_crps(
                torch.from_numpy(y_true),
                torch.from_numpy(y_pred),
                torch.from_numpy(eval_points),
            )
            self.metrics["mis"] = calc_mis(
                torch.from_numpy(y_true), torch.from_numpy(y_pred)
            )

            # deterministic metric
            y_pred = np.mean(y_pred, axis=1)  # # (B, T_p, V, D)

            (
                self.metrics["mae"],
                self.metrics["mse"],
                self.metrics["rmse"],
                self.metrics["rmsle"],
                self.metrics["mape"],
                self.metrics["mspe"],
            ) = self.calc_metric(y_true, y_pred)

        if isinstance(loss, float):
            self.metrics["loss"] = loss
        else:
            self.metrics["loss"] = loss.item()
        self.metrics["time"] = time.time() - self.time_start

    def update_metrics_list(self):

        self.loss_list.append(self.metrics["loss"])
        if self.task in ["forecast", "pretrain"]:
            self.mae_list.append(self.metrics["mae"])
            self.mse_list.append(self.metrics["mse"])
            self.rmse_list.append(self.metrics["rmse"])
            self.rmsle_list.append(self.metrics["rmsle"])
            self.mape_list.append(self.metrics["mape"])
            self.mspe_list.append(self.metrics["mspe"])
            self.crps_list.append(self.metrics["crps"])
            self.mis_list.append(self.metrics["mis"])

    def clear_list(self):
        if self.task in ["forecast", "pretrain"]:
            self.loss_list.clear()
            self.mae_list.clear()
            self.mse_list.clear()
            self.rmse_list.clear()
            self.rmsle_list.clear()
            self.mape_list.clear()
            self.mspe_list.clear()
            self.crps_list.clear()
            self.mis_list.clear()
        elif self.task in ["anomaly", "locate"]:
            self.f1_list.clear()
            self.rec_list.clear()
            self.pre_list.clear()
            self.hr_1_list.clear()
            self.ndcg_1_list.clear()
            self.hr_3_list.clear()
            self.ndcg_3_list.clear()
            self.hr_5_list.clear()
            self.ndcg_5_list.clear()
            self.avg_hr_3_list.clear()
            self.avg_hr_3_scores_list.clear()

    def calc_mean_for_list(self):
        if self.task in ["forecast", "pretrain"]:
            self.mean_metrics["loss"] = np.mean(self.loss_list)
            self.mean_metrics["mae"] = np.mean(self.mae_list)
            self.mean_metrics["mse"] = np.mean(self.mse_list)
            self.mean_metrics["rmse"] = np.mean(self.rmse_list)
            self.mean_metrics["rmsle"] = np.mean(self.rmsle_list)
            self.mean_metrics["mape"] = np.mean(self.mape_list)
            self.mean_metrics["mspe"] = np.mean(self.mspe_list)
            self.mean_metrics["crps"] = np.mean(self.crps_list)
            self.mean_metrics["mis"] = np.mean(self.mis_list)
        elif self.task in ["anomaly", "locate"]:
            pass

    def update_best_metrics(self, epoch=0):
        if self.task in ["forecast", "pretrain"]:
            self.best_metrics["mae"], mae_state = self.get_best_metric(
                self.best_metrics["mae"], self.metrics["mae"]
            )
            self.best_metrics["mse"], mse_state = self.get_best_metric(
                self.best_metrics["mse"], self.metrics["mse"]
            )
            self.best_metrics["rmse"], rmse_state = self.get_best_metric(
                self.best_metrics["rmse"], self.metrics["rmse"]
            )
            self.best_metrics["rmsle"], rmse_state = self.get_best_metric(
                self.best_metrics["rmsle"], self.metrics["rmsle"]
            )
            self.best_metrics["mape"], mape_state = self.get_best_metric(
                self.best_metrics["mape"], self.metrics["mape"]
            )
            self.best_metrics["mspe"], mape_state = self.get_best_metric(
                self.best_metrics["mspe"], self.metrics["mspe"]
            )
            self.best_metrics["crps"], crps_state = self.get_best_metric(
                self.best_metrics["crps"], self.metrics["crps"]
            )
            self.best_metrics["mis"], mis_state = self.get_best_metric(
                self.best_metrics["mis"], self.metrics["mis"]
            )
            if mae_state:
                self.best_metrics["epoch"] = int(epoch)
        elif self.task in ["anomaly", "locate"]:
            self.best_metrics["f1"], f1_state = self.get_best_metric(
                self.best_metrics["f1"], self.metrics["f1"]
            )
            self.best_metrics["rec"], rec_state = self.get_best_metric(
                self.best_metrics["rec"], self.metrics["rec"]
            )
            self.best_metrics["pre"], pre_state = self.get_best_metric(
                self.best_metrics["pre"], self.metrics["pre"]
            )
            self.best_metrics["hr_1"], hr_1_state = self.get_best_metric(
                self.best_metrics["hr_1"], self.metrics["hr_1"]
            )
            self.best_metrics["ndcg_1"], ndcg_1_state = self.get_best_metric(
                self.best_metrics["ndcg_1"], self.metrics["ndcg_1"]
            )
            self.best_metrics["hr_3"], hr_3_state = self.get_best_metric(
                self.best_metrics["hr_3"], self.metrics["hr_3"]
            )
            self.best_metrics["ndcg_3"], ndcg_3_state = self.get_best_metric(
                self.best_metrics["ndcg_3"], self.metrics["ndcg_3"]
            )
            self.best_metrics["hr_5"], hr_5_state = self.get_best_metric(
                self.best_metrics["hr_5"], self.metrics["hr_5"]
            )
            self.best_metrics["ndcg_5"], ndcg_5_state = self.get_best_metric(
                self.best_metrics["ndcg_5"], self.metrics["ndcg_5"]
            )
            self.best_metrics["avg_hr_3"], avg_hr_3_state = self.get_best_metric(
                self.best_metrics["avg_hr_3"], self.metrics["avg_hr_3"]
            )
            self.best_metrics["avg_hr_3_scores"], avg_hr_3_scores_state = (
                self.get_best_metric(
                    self.best_metrics["avg_hr_3_scores"],
                    self.metrics["avg_hr_3_scores"],
                )
            )
            if f1_state:
                self.best_metrics["epoch"] = int(epoch)

    @staticmethod
    def calc_metric(y_true, y_pred):
        mae = MAE(y_true, y_pred)
        mse = MSE(y_true, y_pred)
        rmse = RMSE(y_true, y_pred)
        rmsle = RMSLE(y_true, y_pred)
        mape = MAPE(y_true, y_pred)
        mspe = MSPE(y_true, y_pred)

        return mae, mse, rmse, rmsle, mape, mspe

    def calc_precision_rank_metric(self):
        TP = self.TP
        FP = self.FP
        FN = self.FN
        hrs = self.hrs
        ndcgs = self.ndcgs

        pos = TP + FN
        f1 = TP * 2.0 / (TP + FP + pos) if (TP + FP + pos) > 0 else 0
        rec = TP * 1.0 / pos if pos > 0 else 0
        pre = TP * 1.0 / (TP + FP) if (TP + FP) > 0 else 0

        if hrs is not None:
            hr_1 = hrs[0] * 1.0 / pos
            ndcg_1 = ndcgs[0] * 1.0 / pos
            hr_3 = hrs[2] * 1.0 / pos
            ndcg_3 = ndcgs[2] * 1.0 / pos
            hr_5 = hrs[4] * 1.0 / pos
            ndcg_5 = ndcgs[4] * 1.0 / pos
            avg_hr_3 = (hr_1 + hr_3 + hr_5) / 3
            avg_hr_3_scores = sum(self.avg_hr_3_scores_list) / len(
                self.avg_hr_3_scores_list
            )

        else:
            hr_1, ndcg_1, hr_3, ndcg_3, hr_5, ndcg_5 = 0, 0, 0, 0, 0, 0
            avg_hr_3, avg_hr_3_scores = 0, 0

        self.f1_list.append(f1)
        self.rec_list.append(rec)
        self.pre_list.append(pre)
        self.hr_1_list.append(hr_1)
        self.ndcg_1_list.append(ndcg_1)
        self.hr_3_list.append(hr_3)
        self.ndcg_3_list.append(ndcg_3)
        self.hr_5_list.append(hr_5)
        self.ndcg_5_list.append(ndcg_5)
        self.avg_hr_3_list.append(avg_hr_3)

        if hrs is not None:
            self.metrics["avg_hr_3"] = (
                self.metrics["hr_1"] + self.metrics["hr_3"] + self.metrics["hr_5"]
            ) / 3
        else:
            self.metrics["avg_hr_3"] = None

        return (
            f1,
            rec,
            pre,
            hr_1,
            ndcg_1,
            hr_3,
            ndcg_3,
            hr_5,
            ndcg_5,
            avg_hr_3,
            avg_hr_3_scores,
        )

    def get_best_metric(self, best, candidate):
        state = False
        if self.task == "forecast":
            if candidate < best:
                best = candidate
                state = True
        else:
            if candidate > best:
                best = candidate
                state = True
        return best, state

    def __str__(self):
        """For print"""
        if self.task == "forecast":
            return (
                f"{self.metrics['mae']:<7.4f}, {self.metrics['mse']:<7.4f}, {self.metrics['rmse']:<7.4f}, "
                f"{self.metrics['rmsle']:<7.4f}, {self.metrics['mape']:<7.4f}, {self.metrics['mspe']:<7.4f}, "
                f"{self.metrics['crps']:<7.4f}, {self.metrics['mis']:<7.4f} | {self.best_metrics['epoch'] + 1:<4} "
            )
        elif self.task in ["anomaly", "locate"]:
            return (
                f"{self.metrics['f1']:<7.4f}, {self.metrics['rec']:<7.4f}, {self.metrics['pre']:<7.4f}, "
                f"{self.metrics['hr_1']:<7.4f}, {self.metrics['ndcg_1']:<7.4f}, {self.metrics['hr_3']:<7.4f}, "
                f"{self.metrics['ndcg_3']:<7.4f}, {self.metrics['hr_5']:<7.4f}, {self.metrics['ndcg_5']:<7.4f} | "
                f"{self.best_metrics['epoch'] + 1:<4} "
            )

    def best_str(self):
        """For save"""
        if self.task == "forecast":
            return (
                f"{self.best_metrics['epoch']}, {self.best_metrics['mae']:.4f}, {self.best_metrics['mse']:.4f}, "
                f"{self.best_metrics['rmse']:.4f}, {self.best_metrics['rmsle']:.4f}, {self.best_metrics['mape']:.4f}, "
                f"{self.best_metrics['mspe']:.4f}, {self.best_metrics['crps']:.4f}, {self.best_metrics['mis']:.4f}"
            )
        elif self.task in ["anomaly", "locate"]:
            return (
                f"{self.best_metrics['epoch']}, {self.best_metrics['f1']:.4f}, {self.best_metrics['rec']:.4f}, "
                f"{self.best_metrics['pre']:.4f}, {self.best_metrics['hr_1']:.4f}, {self.best_metrics['ndcg_1']:.4f}, "
                f"{self.best_metrics['hr_3']:.4f}, {self.best_metrics['ndcg_3']:.4f}, {self.best_metrics['hr_5']:.4f}, "
                f"{self.best_metrics['ndcg_5']:.4f}, {self.best_metrics['avg_hr_3']:.4f}, {self.best_metrics['avg_hr_3_scores']:.4f}"
            )

    def get_metrics_dict(self):
        return self.metrics

    def format_metrics(self, target):
        l = []
        if self.task == "forecast":
            if isinstance(target, dict):
                for key in [
                    "loss",
                    "mae",
                    "mse",
                    "rmse",
                    "rmsle",
                    "mape",
                    "mspe",
                    "crps",
                    "mis",
                ]:
                    l.append(target[key])
            elif isinstance(target, list):
                pass
        elif self.task in ["anomaly", "locate"]:
            if isinstance(target, dict):
                for key in [
                    "f1",
                    "rec",
                    "pre",
                    "hr_1",
                    "ndcg_1",
                    "hr_3",
                    "ndcg_3",
                    "hr_5",
                    "ndcg_5",
                    "avg_hr_3",
                    "avg_hr_3_scores",
                ]:
                    l.append(target[key])
            elif isinstance(target, list):
                pass


if __name__ == "__main__":
    m = Metrics()
    y_true = np.random.rand(2, 3, 4, 5)
    y_pred = np.random.rand(2, 1, 3, 4, 5)

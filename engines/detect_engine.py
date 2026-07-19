"""
Training engine for the Stage 2 anomaly detection task.

The engine uses the downstream M3RL system representation for binary
system-level anomaly classification.
"""

import sys
import os


sys.path.append(os.path.dirname(sys.path[0]))
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.amp import GradScaler, autocast
from models.M3RL.stage2 import M3RLStage2
import utils.utils
import utils.tools
from metrics.metrics import Metrics
from utils.input_masking import *


class trainer(nn.Module):
    """Wrap downstream anomaly-detection optimization and evaluation."""

    def __init__(self, settings, log, device):
        super(trainer, self).__init__()

        self.model_dict = {
            "M3RLStage2": M3RLStage2,
            "M3RL": M3RLStage2,
        }
        self.settings = settings
        self.task = settings.task
        self.dataset = settings.dataset
        self.model_name_str = settings.model
        self.model_name = self.model_dict[settings.model]
        if device == torch.device("cpu"):
            self.model = self.model_name(settings, device)
        else:
            self.model = self.model_name(settings, device).cuda()

        self.iters_to_accumulate = settings.iters_to_accumulate
        self.batch_size = settings.batch_size
        warmup_steps = int(10000 / (self.batch_size * self.iters_to_accumulate))

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model).cuda()
        else:
            self.model = self.model.to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

        self.lr_decay = settings.lr_decay
        if self.lr_decay:
            utils.utils.log_string(log, "Applying Lambda Learning rate decay.")
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda epoch: settings.lr_decay_rate**epoch,
            )
        self.if_warmup = settings.if_warmup
        if self.if_warmup:
            self.warmup_scheduler = utils.tools.WarmupLR(
                self.optimizer,
                warmup_steps=warmup_steps,
                iters_to_accumulate=self.iters_to_accumulate,
                init_lr=settings.learning_rate,
            )

        self.scaler = GradScaler()

        if self.model_name_str in ["M3RL"]:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        self.loss.cuda()

        self.clip = settings.max_grad_norm

        self.if_amp = settings.if_amp

        utils.utils.log_string(
            log,
            "Model trainable parameters: {:,}".format(
                utils.utils.count_parameters(self.model)
            ),
        )

        utils.utils.init_seed(seed=42)

    def train_model(
        self,
        x,
        y,
        x_mark_enc,
        x_mark_dec,
        labels,
        adj_x,
        adj_y,
        instances_index_vm,
        loss_epoch,
        ix,
        iters_to_accumulate,
        len_train,
        learning_rate,
        epoch,
        metrics: Metrics,
    ):
        """
        x: B, T, N, F
        y: B, T, N, F
        """
        B, _, _, _ = x[0].shape
        if isinstance(x, tuple):
            x_m = x[0]
        else:
            x_m = x
        if self.model_name_str in ["M3RL"]:
            observed_mask, cond_mask, target_mask = mask_input(
                x_m, y, self.task_name, self.predict_steps, self.mix_masking_strategy
            )
        self.model.train()
        if self.if_amp:
            with autocast():
                y_detect = torch.zeros((B,)).to(x_m.device)
                if labels.dim() == 1:
                    for i in range(B):
                        y_detect[i] = int(labels[i] > -1)
                else:
                    y_detect = labels.any(dim=1).int()

                if self.model_name_str in ["M3RL"]:
                    eps, eps_theta = self.model(
                        x, y, (observed_mask, cond_mask, labels, adj_x)
                    )
                    noise_labels = eps * target_mask
                    predict = eps_theta * target_mask
                elif self.model_name_str in [
                    "Eadro",
                    "AnoFusion",
                ]:
                    predict = self.model(x, adj_x)
                else:
                    predict = self.model(x, adj_x, mask=None)

                if self.model_name_str in ["M3RL"]:
                    loss = self.loss(predict, noise_labels)
                else:
                    loss = self.loss(predict["detect_logits"], y_detect.long())
                loss.requires_grad_(True)
                loss_epoch += loss.item() * len(x)
                loss = loss / iters_to_accumulate
                metrics.update_metrics(labels, predict, loss)

            self.scaler.scale(loss).backward()

            if ((ix + 1) % iters_to_accumulate == 0) or ((ix + 1) == len_train):
                self.scaler.unscale_(self.optimizer)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.if_warmup and (self.warmup_scheduler.stop_flag == False):
                    self.warmup_scheduler.step()

        else:
            y_detect = torch.zeros((B,)).to(x_m.device)
            if labels.dim() == 1:
                for i in range(B):
                    y_detect[i] = int(labels[i] > -1)
            else:
                y_detect = labels.any(dim=1).int()

            if self.model_name_str in ["M3RL"]:
                predict = self.model(x, y, (observed_mask, cond_mask, labels, adj_x))
            elif self.model_name_str in [
                "Eadro",
                "AnoFusion",
            ]:
                predict = self.model(x, adj_x)
            else:
                predict = self.model(x, adj_x, mask=None)

            if self.model_name_str in ["M3RL"]:
                loss = self.loss(predict, noise_labels)
            else:
                loss = self.loss(predict["detect_logits"], y_detect.long())

            loss.requires_grad_(True)
            loss_epoch += loss.item() * len(x)
            loss = loss / iters_to_accumulate

            loss_start_time = time.time()
            loss.backward()
            loss_end_time = time.time()
            print("Loss backward time is {}s.".format(loss_end_time - loss_start_time))

            if ((ix + 1) % iters_to_accumulate == 0) or ((ix + 1) == len_train):
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.if_warmup and (self.warmup_scheduler.stop_flag == False):
                    self.warmup_scheduler.step()
            metrics.update_metrics(labels, predict, loss)

        return metrics, loss_epoch

    def eval_model(
        self,
        x,
        y,
        x_mark_enc,
        x_mark_dec,
        labels,
        adj_x,
        adj_y,
        instances_index_vm,
        metrics,
        epoch,
    ):
        """
        input_data: B, T, N, F
        real_val: B, T, N , F
        """
        B, _, _, _ = x[0].shape
        if isinstance(x, tuple):
            x_m = x[0]
        else:
            x_m = x
        if self.model_name_str in ["M3RL"]:
            observed_mask, cond_mask, target_mask = mask_input(
                x_m, y, self.task_name, self.predict_steps, self.mix_masking_strategy
            )
        self.model.eval()
        with torch.no_grad():
            if self.model_name_str in ["M3RL"]:
                predict = self.model(x, y, (observed_mask, cond_mask, labels, adj_x))
            elif self.model_name_str in [
                "Eadro",
                "AnoFusion",
            ]:
                predict = self.model(x, adj_x)
            else:
                predict = self.model(x, adj_x, mask=None)
        loss = 0.0
        metrics.update_metrics(labels, predict, loss)
        metrics.update_best_metrics(epoch)
        return metrics

    def test_model(
        self,
        x,
        y,
        x_mark_enc,
        x_mark_dec,
        labels,
        adj_x,
        adj_y,
        instances_index_vm,
        metrics,
        epoch,
        samples,
        targets,
    ):
        B, _, _, _ = x[0].shape
        if isinstance(x, tuple):
            x_m = x[0]
        else:
            x_m = x
        if self.model_name_str in ["M3RL"]:
            observed_mask, cond_mask, target_mask = mask_input(
                x_m, y, self.task_name, self.predict_steps, self.mix_masking_strategy
            )
        self.model.eval()
        with torch.no_grad():
            if self.model_name_str in ["M3RL"]:
                predict = self.model(x, y, (observed_mask, cond_mask, labels, adj_x))
            elif self.model_name_str in [
                "Eadro",
                "AnoFusion",
            ]:
                predict = self.model(x, adj_x)
            else:
                predict = self.model(x, adj_x, mask=None)

            loss = 0.0
            metrics.update_metrics(labels, predict, loss)
            metrics.update_best_metrics(epoch)
        return metrics

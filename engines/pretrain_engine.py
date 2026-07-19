"""
Training engine for M3RL Stage 1.

This module contains the optimization loop for the pre-training model and the
weight-loading helper used when continuing Stage 1 or initializing Stage 2.
The forward/loss computation remains in `models/M3RL/stage1.py`.
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
from models.M3RL.stage1 import M3RLStage1
from models.M3RL.stage2 import M3RLStage2
import utils.utils
import utils.tools
from metrics.metrics import Metrics
from utils.input_masking import *
from utils.masking import DynamicMaskingManager


class trainer(nn.Module):
    """Wrap model construction, optimization, and evaluation for Stage 1."""

    def __init__(self, settings, log, device):
        super(trainer, self).__init__()

        self.model_dict = {
            "M3RLStage1": M3RLStage1,
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

        if self.model_name_str in ["M3RLStage1"]:
            self.loss = torch.nn.MSELoss()
            self.loss.cuda()

        else:
            if self.dataset == "Germany":
                self.loss1 = torch.nn.CrossEntropyLoss()
                self.loss2 = torch.nn.BCEWithLogitsLoss()
                self.loss1.cuda()
                self.loss2.cuda()
            else:
                self.loss1 = torch.nn.CrossEntropyLoss()
                self.loss2 = torch.nn.CrossEntropyLoss(ignore_index=-1)
                self.loss1.cuda()
                self.loss2.cuda()

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
        x: tuple, x[0] is metric, x[1] is log, x[2] is trace
        """
        masking_manager = DynamicMaskingManager()
        x_mask = masking_manager.forward(x)
        B, _, N, _ = x[0].shape
        if isinstance(x, tuple):
            x_m = x[0]
        else:
            x_m = x
        y_locate_prob = torch.zeros((B, N)).to(x_m.device)
        if labels.dim() == 1:
            for i in range(B):
                if labels[i] > -1:
                    y_locate_prob[i, labels[i]] = 1
        else:
            y_locate_prob = labels.copy()
        y_detect = torch.zeros((B,)).to(x_m.device)
        if labels.dim() == 1:
            for i in range(B):
                y_detect[i] = int(labels[i] > -1)
        else:
            y_detect = labels.any(dim=1).int()
        #     observed_mask, cond_mask, target_mask = mask_input(x_m, y, self.task_name, self.predict_steps,
        self.model.train()
        if self.if_amp:
            with autocast():
                y_detect = torch.zeros((B,)).to(x_m.device)
                if labels.dim() == 1:
                    for i in range(B):
                        y_detect[i] = int(labels[i] > -1)
                else:
                    y_detect = labels.any(dim=1).int()

                if self.model_name_str in ["M3RLStage1"]:
                    fuse_out, fuse_in, _ = self.model(x[0], x[1], x[2], adj_x)
                    target = fuse_in
                    predict = fuse_out
                else:
                    predict = self.model(x, adj_x, mask=None)

                if self.model_name_str in ["M3RLStage1"]:
                    loss = self.loss(predict, target)
                else:
                    loss1 = self.loss1(predict["detect_logits"], y_detect.long())
                    if self.dataset == "Germany":
                        loss2 = self.loss2(predict["locate_logits"], labels.float())
                    else:
                        loss2 = self.loss2(predict["locate_logits"], labels.long())
                    loss = 0.5 * loss1 + 0.5 * loss2

                loss.requires_grad_(True)
                loss_epoch += loss.item() * B
                loss = loss / iters_to_accumulate
                metrics.update_metrics(target, predict, loss)

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
            if self.model_name_str in ["M3RLStage1"]:
                reconstructions, embs, _ = self.model(
                    x, x_mask, adj_x, y, y_detect, y_locate_prob
                )
                target = embs["metric"]
                predict = reconstructions["metric"]
            else:
                predict = self.model(x, adj_x, mask=None)

            if self.model_name_str in ["M3RLStage1"]:
                loss = self.model.loss_function(reconstructions, epoch)
            else:
                loss1 = self.loss1(predict["detect_logits"], y_detect.long())
                if self.dataset == "Germany":
                    loss2 = self.loss2(predict["locate_logits"], labels.float())
                else:
                    loss2 = self.loss2(predict["locate_logits"], labels.long())
                loss = 0.5 * loss1 + 0.5 * loss2

            loss.requires_grad_(True)
            loss_epoch += loss.item() * B
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

            target = target.detach().cpu().numpy()
            predict = predict.detach().cpu().numpy()
            metrics.update_metrics(target, predict, loss)

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
        masking_manager = DynamicMaskingManager()
        x_mask = masking_manager.forward(x)
        B, _, N, _ = x[0].shape
        if isinstance(x, tuple):
            x_m = x[0]
        else:
            x_m = x

        self.model.eval()
        y_locate_prob = torch.zeros((B, N)).to(x_m.device)
        if labels.dim() == 1:
            for i in range(B):
                if labels[i] > -1:
                    y_locate_prob[i, labels[i]] = 1
        else:
            y_locate_prob = labels.copy()
        y_detect = torch.zeros((B,)).to(x_m.device)
        if labels.dim() == 1:
            for i in range(B):
                y_detect[i] = int(labels[i] > -1)
        else:
            y_detect = labels.any(dim=1).int()
        with torch.no_grad():
            if self.model_name_str in ["M3RLStage1"]:
                reconstructions, embs, _ = self.model(
                    x, x_mask, adj_x, y, y_detect, y_locate_prob
                )
                loss = self.model.loss_function(reconstructions, epoch)
                target = embs["metric"]
                predict = reconstructions["metric"]

            else:
                predict = self.model(x, adj_x, mask=None)

        target = target.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()
        metrics.update_metrics(target, predict, loss)
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
        masking_manager = DynamicMaskingManager()
        x_mask = masking_manager.forward(x)
        B, _, N, _ = x[0].shape
        if isinstance(x, tuple):
            x_m = x[0]
        else:
            x_m = x
        #     observed_mask, cond_mask, target_mask = mask_input(x_m, y, self.task_name, self.predict_steps,
        self.model.eval()
        y_locate_prob = torch.zeros((B, N)).to(x_m.device)
        if labels.dim() == 1:
            for i in range(B):
                if labels[i] > -1:
                    y_locate_prob[i, labels[i]] = 1
        else:
            y_locate_prob = labels.copy()
        y_detect = torch.zeros((B,)).to(x_m.device)
        if labels.dim() == 1:
            for i in range(B):
                y_detect[i] = int(labels[i] > -1)
        else:
            y_detect = labels.any(dim=1).int()
        with torch.no_grad():
            if self.model_name_str in ["M3RLStage1"]:
                reconstructions, embs, system_vector = self.model(
                    x, x_mask, adj_x, y, y_detect, y_locate_prob
                )
                loss = self.model.loss_function(reconstructions, epoch)
                target = embs["metric"]
                predict = reconstructions["metric"]

            elif self.model_name_str in [
                "Eadro",
                "AnoFusion",
                "DiagFusion",
                "Hades",
                "SCWarn",
                "ART",
                "MRCA",
                "Medicine",
                "UACAD",
                "LogAnomaly",
                "TraceAnomaly",
                "MADCMC",
                "Dejavu",
            ]:
                predict = self.model(x, adj_x)
                system_vector = None
            else:
                predict = self.model(x, adj_x, mask=None)

            loss = 0.0
            target = target.detach().cpu().numpy()
            predict = predict.detach().cpu().numpy()
            metrics.update_metrics(target, predict, loss)
            metrics.update_best_metrics(epoch)
        return metrics, system_vector


def load_pretrained_weights(
    model, stage: int, pretrained_path: str, device: torch.device
):
    """Load Stage 1 weights into a Stage 1 continuation or Stage 2 model."""
    print(f"Loading pretrained weights from: {pretrained_path}")
    try:
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()

        if stage == 1:
            if pretrained_path.find("M3RLStage1") != -1:
                pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items()}
        elif stage == 2:
            pretrained_dict_filtered = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict
                and (
                    k.startswith("embedding.")
                    or k.startswith("input_projections.")
                    or k.startswith("context_k_projections.")
                    or k.startswith("context_v_projections.")
                    or k.startswith("restoration_blocks.")
                    or k.startswith("output_projections.")
                )
            }
        else:
            pretrained_dict_filtered = {}

        if not pretrained_dict_filtered:
            print(
                "Warning: No matching layers found in pretrained weights file for loading."
            )
            return

        print(f"Found {len(pretrained_dict_filtered)} parameter modules to load.")

        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict)
        print("Successfully loaded pretrained weights.")

    except FileNotFoundError:
        print(
            f"Error: Pretrained weights file not found at {pretrained_path}. Skipping loading."
        )
    except Exception as e:
        print(f"Error loading pretrained weights: {e}. Skipping loading.")

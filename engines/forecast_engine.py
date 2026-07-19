"""
Training engine for the Stage 2 forecasting task.

The engine initializes `M3RLStage2`, optionally separates pretrained modules
into a lower-learning-rate parameter group, and computes forecasting losses and
metrics without changing the model internals.
"""

import sys
import os

import torch.autograd

sys.path.append(os.path.dirname(sys.path[0]))
import time
import random
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler as GradScaler, autocast
from models.M3RL.stage2 import M3RLStage2
import utils.utils
import utils.tools
from metrics.metrics import Metrics
from utils.input_masking import *


class trainer(nn.Module):
    """Wrap downstream forecasting optimization and evaluation."""

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

        if self.model_name_str in ["M3RLStage2"]:
            finetune_lr = settings.learning_rate / 10.0
            finetune_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("embedding"):
                    print(
                        f"Parameter '{name}' assigned to finetune group with LR: {finetune_lr}"
                    )
                    finetune_params.append(param)
                elif name.startswith("fusion"):
                    print(
                        f"Parameter '{name}' assigned to finetune group with LR: {finetune_lr}"
                    )
                    finetune_params.append(param)
                elif name.startswith("encoder"):
                    print(
                        f"Parameter '{name}' assigned to finetune group with LR: {finetune_lr}"
                    )
                    finetune_params.append(param)
                else:
                    other_params.append(param)
            if len(list(self.model.parameters())) != len(finetune_params) + len(
                other_params
            ):
                print(
                    "Warning: Some parameters are not assigned to any optimizer group!"
                )
            param_groups = [
                {"params": other_params, "lr": settings.learning_rate},
                {"params": finetune_params, "lr": finetune_lr},
            ]
            self.optimizer = optim.Adam(
                param_groups, weight_decay=settings.weight_decay
            )
        else:
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

        if self.model_name_str in ["M3RLStage2"]:
            self.loss = torch.nn.MSELoss()
            self.loss1 = torch.nn.CrossEntropyLoss()
            self.loss2 = torch.nn.CrossEntropyLoss(ignore_index=-1)
            self.loss.cuda()
            self.loss1.cuda()
            self.loss2.cuda()
        else:
            if self.dataset == "Germany":
                self.loss1 = torch.nn.CrossEntropyLoss()
                self.loss2 = torch.nn.BCEWithLogitsLoss()
                self.loss1.cuda()
                self.loss2.cuda()
            else:
                self.loss = torch.nn.MSELoss()
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
        x: B, T, N, F
        y: B, T, N, F
        """
        if isinstance(x, tuple):
            x_m = x[0]
        else:
            x_m = x
        if self.model_name_str in ["MSTDiff", "DiffSTG", "M3RL", "TSDE"]:
            observed_mask, cond_mask, target_mask = mask_input(
                x_m, y, self.task_name, self.predict_steps, self.mix_masking_strategy
            )

        self.model.train()
        if self.if_amp:
            with autocast():
                if self.model_name_str in ["M3RL"]:
                    eps, eps_theta = self.model(
                        x, y, (observed_mask, cond_mask, adj_x), labels
                    )
                    y = eps * target_mask
                    predict = eps_theta * target_mask
                elif self.model_name_str in ["MSTDiff", "DiffSTG", "TSDE", "TSDiff"]:
                    eps, eps_theta = self.model(
                        x_m, y, (observed_mask, cond_mask, adj_x)
                    )
                    y = eps * target_mask
                    predict = eps_theta * target_mask
                elif self.model_name_str in [
                    "Informer",
                    "Autoformer",
                    "FEDformer",
                    "TimesNet",
                    "PatchTST",
                    "DLinear",
                    "GPT4TS",
                ]:
                    predict = self.model(x_m, y, x_mark_enc, x_mark_dec, mask=None)
                    predict = predict.mean(dim=1)
                elif self.model_name_str in ["STMformer"]:
                    predict, label_predict = self.model(
                        x_m,
                        y,
                        x_mark_enc,
                        x_mark_dec,
                        adj_x,
                        adj_y,
                        instances_index_vm,
                        mask=None,
                    )
                elif self.model_name_str in ["STSGCN", "STSGT", "MAGNN", "FC_STGNN"]:
                    predict = self.model(x_m, adj_x, mask=None)
                    predict = predict.mean(dim=1)
                else:
                    predict = self.model(x_m, adj_x, mask=None)

                loss = self.loss(predict, y)
                loss.requires_grad_(True)

                loss_epoch += loss.item() * len(x_m)
                loss = loss / iters_to_accumulate

            loss_start_time = time.time()
            self.scaler.scale(loss).backward()
            loss_end_time = time.time()
            print(
                "Loss backward time is {:.6f}s.".format(loss_end_time - loss_start_time)
            )

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
            if self.model_name_str in ["M3RLStage2"]:
                predict, _ = self.model(x[0], x[1], x[2], adj_x, self.task)
                predict = predict.mean(dim=1)
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
                predict = predict.mean(dim=1)
            elif self.model_name_str in ["MSTDiff", "DiffSTG", "TSDE", "TSDiff"]:
                eps, eps_theta = self.model(x_m, y, (observed_mask, cond_mask, adj_x))
                y = eps * target_mask
                predict = eps_theta * target_mask
            elif self.model_name_str in [
                "Informer",
                "Autoformer",
                "FEDformer",
                "TimesNet",
                "PatchTST",
                "DLinear",
                "GPT4TS",
            ]:
                predict = self.model(x_m, y, x_mark_enc, x_mark_dec, mask=None)
                predict = predict.mean(dim=1)
            elif self.model_name_str in ["STMformer"]:
                predict, label_predict = self.model(
                    x_m,
                    y,
                    x_mark_enc,
                    x_mark_dec,
                    adj_x,
                    adj_y,
                    instances_index_vm,
                    mask=None,
                )
            elif self.model_name_str in ["STSGCN", "STSGT", "MAGNN", "FC_STGNN"]:
                predict = self.model(x_m, adj_x, mask=None)
                predict = predict.mean(dim=1)
            else:
                predict = self.model(x_m, adj_x, mask=None)

            loss = self.loss(predict, y)
            loss.requires_grad_(True)

            loss_epoch += loss.item() * len(x_m)
            loss = loss / iters_to_accumulate

            loss_start_time = time.time()
            loss.backward()
            loss_end_time = time.time()
            print(
                "Loss backward time is {:.6f}s.".format(loss_end_time - loss_start_time)
            )

            if ((ix + 1) % iters_to_accumulate == 0) or ((ix + 1) == len_train):
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.if_warmup and (self.warmup_scheduler.stop_flag == False):
                    self.warmup_scheduler.step()
        y = y.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()
        metrics.update_metrics(y, predict, loss)

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
        if isinstance(x, tuple):
            x_m = x[0]
        else:
            x_m = x

        if self.model_name_str in ["MSTDiff", "DiffSTG", "TSDE"]:
            observed_mask, cond_mask, target_mask = mask_input(
                x_m, y, self.task_name, self.predict_steps, self.mix_masking_strategy
            )

        self.model.eval()
        with torch.no_grad():
            if self.model_name_str in ["M3RLStage2"]:
                predict, _ = self.model(x[0], x[1], x[2], adj_x, self.task)
                predict = predict.mean(dim=1)
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
                predict = predict.mean(dim=1)
            elif self.model_name_str in ["MSTDiff", "DiffSTG", "TSDE", "TSDiff"]:
                loss_sum = 0
                k = 1
                eval_sequence = random.sample(self.sequence, k)
                for i in eval_sequence:
                    eps, eps_theta = self.model(
                        x_m,
                        y,
                        (observed_mask, cond_mask, adj_x),
                        set_t=i,
                        is_train=False,
                    )
                    gt = eps * target_mask
                    predict = eps_theta * target_mask
                    loss_per_step = self.loss(predict, gt).detach()
                    gt = gt.detach().cpu().numpy()
                    predict = predict.detach().cpu().numpy()
                    metrics.update_metrics(gt, predict, loss_per_step)
                    loss_sum += loss_per_step
                loss_avg = loss_sum / len(eval_sequence)
            elif self.model_name_str in [
                "Informer",
                "Autoformer",
                "FEDformer",
                "TimesNet",
                "PatchTST",
                "DLinear",
                "GPT4TS",
            ]:
                predict = self.model(
                    x_m, y, x_mark_enc, x_mark_dec, mask=None
                )  # B n T N F
                predict = predict.mean(dim=1)  # B T N F
            elif self.model_name_str in ["STMformer"]:
                predict, label_predict = self.model(
                    x_m,
                    y,
                    x_mark_enc,
                    x_mark_dec,
                    adj_x,
                    adj_y,
                    instances_index_vm,
                    mask=None,
                )
            elif self.model_name_str in ["STSGCN", "STSGT", "MAGNN", "FC_STGNN"]:
                predict = self.model(x_m, adj_x, mask=None)
                predict = predict.mean(dim=1)  # B T N F
            else:
                predict = self.model(x_m, adj_x, mask=None)

            if self.model_name_str in ["MSTDiff", "DiffSTG", "TSDE", "TSDiff"]:
                loss = loss_avg
            else:
                loss = self.loss(predict, y).detach()

        if self.model_name_str not in ["MSTDiff", "DiffSTG", "TSDE", "TSDiff"]:
            y = y.detach().cpu().numpy()
            predict = predict.detach().cpu().numpy()
            metrics.update_metrics(y, predict, loss)
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
        if isinstance(x, tuple):
            x_m = x[0]
        else:
            x_m = x
        if self.model_name_str in ["MSTDiff", "DiffSTG", "M3RL", "TSDE"]:
            observed_mask, cond_mask, target_mask = mask_input(
                x_m, y, self.task_name, self.predict_steps, self.mix_masking_strategy
            )

        self.model.eval()
        with torch.no_grad():
            if self.model_name_str in ["M3RLStage2"]:
                predict, system_vector = self.model(x[0], x[1], x[2], adj_x, self.task)
                predict = predict.mean(dim=1)
                full_predict = torch.cat((x_m, predict), dim=1).to(
                    x_m.device
                )  # B T N F
                full_ground = torch.cat((x_m, y), dim=1).to(x_m.device)  # B T N F
                predict = predict[:, -self.settings.predict_steps :, :, :]

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
                predict = predict.mean(dim=1)
                full_predict = torch.cat((x_m, predict), dim=1).to(
                    x_m.device
                )  # B T N F
                full_ground = torch.cat((x_m, y), dim=1).to(x_m.device)  # B T N F
                predict = predict[:, -self.settings.predict_steps :, :, :]
                system_vector = None
            elif self.model_name_str in ["MSTDiff", "DiffSTG", "TSDE", "TSDiff"]:
                predict = self.model.predict(
                    x_m,
                    y,
                    (observed_mask, cond_mask, adj_x),
                    n_samples=self.settings.n_samples,
                )
                if self.model_name_str in ["MSTDiff", "DiffSTG", "TSDE", "TSDiff"]:
                    full_predict = predict  # B n_samples T N F
                    full_ground = torch.cat((x_m, y), dim=1).to(x_m.device)  # B T N F
                    predict = predict[:, :, -self.settings.predict_steps :, :, :]
            elif self.model_name_str in [
                "Informer",
                "Autoformer",
                "FEDformer",
                "TimesNet",
                "PatchTST",
                "DLinear",
                "GPT4TS",
            ]:
                predict = self.model(
                    x_m, y, x_mark_enc, x_mark_dec, mask=None
                )  # # B n T_p N F
                prev_samples = x_m.unsqueeze(dim=1).repeat(
                    1, self.settings.n_samples, 1, 1, 1
                )  # B n T_h N F
                full_predict = torch.cat((prev_samples, predict), dim=2).to(
                    x_m.device
                )  # B n T N F
                full_ground = torch.cat((x_m, y), dim=1).to(x_m.device)  # B T N F
            elif self.model_name_str in ["STMformer"]:
                predict, label_predict = self.model(
                    x_m,
                    y,
                    x_mark_enc,
                    x_mark_dec,
                    adj_x,
                    adj_y,
                    instances_index_vm,
                    mask=None,
                )
                full_predict = (
                    torch.cat((x_m, predict), dim=1).unsqueeze(dim=1).to(x_m.device)
                )  # B n T N F
                full_ground = torch.cat((x_m, y), dim=1).to(x_m.device)  # B T N F
            elif self.model_name_str in ["STSGCN", "STSGT", "MAGNN", "FC_STGNN"]:
                predict = self.model(x_m, adj_x, mask=None)  # B n T N F
                prev_samples = x_m.unsqueeze(dim=1).repeat(
                    1, self.settings.n_samples, 1, 1, 1
                )  # B n T/2 N F
                full_predict = torch.cat((prev_samples, predict), dim=2).to(
                    x_m.device
                )  # B n T N F
                full_ground = torch.cat((x_m, y), dim=1).to(x_m.device)  # B T N F
            else:
                predict = self.model(
                    x_m,
                    y,
                    x_mark_enc,
                    x_mark_dec,
                    adj_x,
                    adj_y,
                    instances_index_vm,
                    mask=None,
                )
                full_predict = (
                    torch.cat((x_m, predict), dim=1).unsqueeze(dim=1).to(x_m.device)
                )  # B 1 T N F
                full_ground = torch.cat((x_m, y), dim=1).to(x_m.device)  # B T N F

            loss = 0.0
            y = y.detach().cpu().numpy()
            predict = predict.detach().cpu().numpy()
            metrics.update_metrics(y, predict, loss)
            metrics.update_best_metrics(epoch)
            samples.append(full_predict.cpu())
            targets.append(full_ground.cpu())

        return metrics, samples, targets, system_vector


def load_pretrained_weights(
    model, stage: int, pretrained_path: str, device: torch.device
):
    """Load Stage 1 pretrained weights into the Stage 2 forecasting model."""
    print(f"Loading pretrained weights from: {pretrained_path}")
    try:
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()

        if stage == 2:
            if pretrained_path.find("M3RLStage1") != -1:
                pretrained_dict_filtered = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict
                    and (
                        k.startswith("embedding.")
                        or k.startswith("fusion.")
                        or k.startswith("encoder_stnet.")
                        or k.startswith("encoder_aggregation.")
                        or k.startswith("input_projections.")
                        or k.startswith("context_k_projections.")
                        or k.startswith("context_v_projections.")
                        or k.startswith("restoration_blocks.")
                        or k.startswith("output_projections.")
                    )
                }
            else:
                pretrained_dict_filtered = {}
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

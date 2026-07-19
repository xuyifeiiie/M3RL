import ast
import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupLR(LRScheduler):
    def __init__(
        self, optimizer, warmup_steps, iters_to_accumulate, init_lr=1e-4, last_epoch=-1
    ):
        """
        optimizer: Optimizer object.
        warmup_steps: Number of warm-up steps.
        iters_to_accumulate: Gradient accumulation factor.
        last_epoch: Last scheduler epoch.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.iters_to_accumulate = iters_to_accumulate
        self.init_lr = init_lr
        self.real_iters = self.warmup_steps * self.iters_to_accumulate
        self.stop_flag = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linearly increase the learning rate during warm-up.
            if self.last_epoch == 0:
                print("Warm up beginning...")
                return [
                    base_lr * (self.last_epoch + 1) / self.real_iters
                    for base_lr in self.base_lrs
                ]
            else:
                return [
                    base_lr
                    * self.iters_to_accumulate
                    * self.last_epoch
                    / self.real_iters
                    for base_lr in self.base_lrs
                ]
        else:
            # After warm-up, reset to the configured initial learning rate.
            self.stop_flag = True
            print("Warm up finished!")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.init_lr
            return [self.init_lr]


class EarlyStopping:
    def __init__(self, task, expid, patience=7, verbose=False, delta=0, save_limit=2):
        self.expid = expid
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_limit = save_limit
        self.task = task

    def __call__(self, val_loss, model, path):
        if self.task in ["forecast", "pretrain"]:
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
            elif score <= self.best_score + self.delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
                self.counter = 0
        elif self.task in ["locate", "detect"]:
            score = val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
            elif score <= self.best_score + self.delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
                self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        files, subfolders = self.get_subfiles_subfolders(path)
        print_val_loss_min = "{:.3f}".format(val_loss)
        for file in files:
            expid_str = file.split("_")[-4]
            if expid_str != str(self.expid):
                files.remove(file)
        if self.task in ["forecast", "pretrain"]:
            if len(files) >= self.save_limit:
                sort_files = sorted(files)
                print("The number of saved model has exceeded the limit.")
                for file in sort_files[(-self.save_limit + 1) :]:
                    print("Begin deleting")
                    os.remove(file)
                    print("{} has been deleted".format(file))
            else:
                pass
            if self.verbose:
                print(
                    f"Validation mae decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                )
            self.val_loss_min = val_loss
            # torch.save(model.state_dict(), path + "/exp_" + str(self.expid) + "_" + str(round(self.val_loss_min, 3)) + "_best_model.pth")
            torch.save(
                model.state_dict(),
                path
                + "/exp_"
                + str(self.expid)
                + "_"
                + print_val_loss_min
                + "_best_model.pth",
            )
        elif self.task in ["locate", "detect"]:
            if len(files) >= self.save_limit:
                sort_files = sorted(files, reverse=True)
                print("The number of saved model has exceeded the limit.")
                for file in sort_files[(-self.save_limit + 1) :]:
                    print("Begin deleting")
                    os.remove(file)
                    print("{} has been deleted".format(file))
            else:
                pass
            if self.verbose:
                print(
                    f"Validation f1 increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
                )
            self.val_loss_min = val_loss
            # torch.save(model.state_dict(), path + "/exp_" + str(self.expid) + "_" + str(round(self.val_loss_min, 3)) + "_best_model.pth")
            torch.save(
                model.state_dict(),
                path
                + "/exp_"
                + str(self.expid)
                + "_"
                + print_val_loss_min
                + "_best_model.pth",
            )

    def get_subfiles_subfolders(self, folder_path):
        """
        Get subfile and subfolders of a directory
        :param folder_path: directory path
        :return: subfile and subfolders path list
        """
        files = []
        subfolders = []
        abs_path = os.path.abspath(folder_path)
        for file in os.listdir(abs_path):
            file_path = os.path.join(abs_path, file)
            if os.path.isdir(file_path):
                subfolders.append(file_path)
            else:
                files.append(file_path)
        return files, subfolders

    def find_best_model(self, save_path):
        files, _ = self.get_subfiles_subfolders(save_path)
        if len(files) == 0:
            print("No saved model found in the directory: {}.".format(save_path))
            return None
        else:
            sort_files = sorted(files)
            if (
                self.task in ["forecast", "pretrain"]
                or save_path.find("M3RLStage1") != -1
            ):
                best_model_file = sort_files[0]
            else:
                best_model_file = sort_files[-1]
            return best_model_file


def check_str_type(v):
    v = v.strip()
    v_o = v
    if len(v) == 0:
        sys.exit(111)
    try:
        tmp = ast.literal_eval(v)
    except ValueError:
        return v_o
    except SyntaxError:
        pass
    else:
        if type(tmp) in [int, float, bool]:
            if tmp in set((True, False)):
                return tmp
            if type(tmp) is int:
                return tmp
            if type(tmp) is float:
                return tmp
        else:
            return ast.literal_eval(v_o)


class SettingsFromArgs:
    def __init__(self, args_dict):
        if isinstance(args_dict, dict):
            for k in args_dict.keys():
                setattr(self, k, args_dict[k])


class SettingsFromConfigs:
    def __init__(self, configs):
        config_dict = {}
        for section in configs.sections():
            tmp_dict = {
                key: check_str_type(value) for key, value in configs[section].items()
            }
            config_dict.update(tmp_dict)
        if isinstance(config_dict, dict):
            for k in config_dict.keys():
                setattr(self, k, config_dict[k])

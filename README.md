# M3RL

Official implementation of **M3RL** for the paper **“Learning general-purpose and robust representations of microservice system states from multi-modal data”**.

M3RL learns reusable representations of microservice system states from metrics, logs, traces, and dynamic service dependencies. The released code follows the current implementation as the source of truth for runnable details, while keeping the two-stage workflow described in the paper:

- **Stage 1 pre-training** learns multi-modal service representations with restoration, future metric prediction, and contrastive objectives.
- **Stage 2 downstream fine-tuning** reuses the pretrained modules for metric forecasting, anomaly detection, and root cause localization.

Repository: [xuyifeiiie/M3RL.git](https://github.com/xuyifeiiie/M3RL.git)

## Environment

The paper experiments used PyTorch 2.5.1 on an NVIDIA A100 80G GPU. A typical deployment is:

```bash
git clone https://github.com/xuyifeiiie/M3RL.git
cd M3RL
conda create -n m3rl python=3.10
conda activate m3rl
pip install -r requirements.txt
```

Some model components depend on PyTorch Geometric packages such as `torch-scatter`, `torch-sparse`, and `torch-geometric`. These packages must match your local PyTorch and CUDA versions, so install them manually following the official PyG installation guide instead of blindly installing all optional requirements. `requirements-optional.txt` records optional packages used by some environments, but it is not intended to be a one-command portable installer.

If `ModuleNotFoundError: torch_scatter` appears, the PyG dependency set is not installed correctly for the active PyTorch/CUDA environment.

## Dataset

The public dataset used in the paper is available at [Zenodo: 10.5281/zenodo.7615394](https://doi.org/10.5281/zenodo.7615394).

The experiment scripts load telemetry from the data root specified by `--datapath`; the default is `../data`. After downloading and extracting the raw telemetry, organize dataset folders under that data root, for example:

```text
../data/
  SN-Eadro Dataset/
  TT-Eadro Dataset/
```

## Repository Tree

```text
M3RL/
  configs/
    SN-Eadro Dataset/      Stage 1/Stage 2 configs for the SN-Eadro dataset
    TT-Eadro Dataset/      Stage 1/Stage 2 configs for the TT-Eadro dataset
  engines/
    pretrain_engine.py     Stage 1 training/evaluation loop
    forecast_engine.py     Stage 2 forecasting loop
    detect_engine.py       Stage 2 anomaly detection loop
    locate_engine.py       Stage 2 root cause localization loop
  exps/
    pretrain.py            Stage 1 command-line entry
    forecast.py            Stage 2 forecasting entry
    detect.py              Stage 2 anomaly detection entry
    locate.py              Stage 2 localization entry
  layers/
    embed.py               Shared embedding layers
    forecast.py            Forecasting heads
  metrics/
    metrics.py             Forecasting, detection, and localization metrics
  models/M3RL/
    stage1.py              M3RL pre-training model
    stage2.py              M3RL downstream model
    embedding.py           Metric/log/trace embedding modules
    fusion.py              Cross-modal fusion layer
    stnet.py               Spatio-temporal encoder backbone
    spatial.py             Dynamic dependency spatial layers
    temporal.py            Temporal modeling layers
    contrast.py            Causal contrastive objective for Stage 1
    common.py              Shared heads and aggregation modules
  utils/
    input_masking.py       Downstream input masking utilities
    masking.py             Stage 1 dynamic multi-modal masking
    perturb.py             Robustness perturbation helpers
    semantics.py           Log semantic feature extraction
    tools.py               Config, scheduler, and early stopping utilities
    utils.py               Dataset loading and preprocessing utilities
```

## Configuration

Experiment settings are stored in:

```text
configs/<dataset config directory>/<model>.conf
```

The retained dataset config directories are:

- `SN-Eadro Dataset`
- `TT-Eadro Dataset`

Use:

- `M3RLStage1.conf` for Stage 1 pre-training.
- `M3RLStage2.conf` for Stage 2 downstream fine-tuning.

The downstream task is controlled by the `[data] task=...` field and should match the selected entry script.

## Running

Run commands from the repository root (`M3RL/`).

### 1. Stage 1 Pre-training

```bash
python exps/pretrain.py --dataset "SN-Eadro Dataset" --model M3RLStage1
```

This trains `models/M3RL/stage1.py` with the configured restoration, prediction, and contrastive objectives.

### 2. Stage 2 Downstream Fine-tuning

Forecasting:

```bash
python exps/forecast.py --dataset "TT-Eadro Dataset" --model M3RLStage2
```

Anomaly detection:

```bash
python exps/detect.py --dataset "SN-Eadro Dataset" --model M3RLStage2
```

Root cause localization:

```bash
python exps/locate.py --dataset "SN-Eadro Dataset" --model M3RLStage2
```

Before running a downstream script, check that the selected `M3RLStage2.conf` has a matching `[data] task` value (`forecast`, `detect`, or `locate`). The experiment scripts read the selected config from `configs/<dataset>/<model>.conf`, so command-line `--dataset` and `--model` must match an existing config directory and filename.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{xu2026m3rl,
  title = {Learning general-purpose and robust representations of microservice system states from multi-modal data},
  author = {Xu, Yifei and Ge, Jingguo and Wu, Yulei and Ma, Yuxiang and Li, Hui and Wu, Bingzhen and Li, Tong},
  journal = {Information Processing and Management},
  volume = {63},
  pages = {104937},
  year = {2026},
  doi = {10.1016/j.ipm.2026.104937}
}
```

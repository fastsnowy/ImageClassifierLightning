# imageclassifierlightning

## Setup
### prerequisites
- CUDA11.7.1
- Weights & Biases(wandb) Account (for logging)
  - https://wandb.ai/
### Install dependencies
Need to install Rye
- [Rye](https://rye-up.com/guide/installation/)

### Install the package
```bash
rye sync --no-lock
```

## Usage

logger select **wandb** or **csv**

if you want to use wandb, you need to login wandb.

### wandb login
- need wandb account
```bash
rye shell
wandb login
```
### config setting
copy default.yaml and rename it and edit paths.
```bash
cp config/dataset/default.yaml config/dataset/yourdataset.yaml
cp config/trainer/default.yaml config/trainer/yourtrainer.yaml
```

1. dataset

**need to edit name, train_path, test_path, aug_path**

> [!NOTE]
> if you don't use augment data, you don't need to edit aug_path (blank is ok)

```yaml
# config/dataset/default.yaml
name: dataset's name
train_path: path/to/train/dataset
test_path: path/to/test/dataset
aug_path: path/to/augmented/dataset
```

2. trainer

**need to edit save_dir and logger**
```yaml
# config/trainer/default.yaml
augment: False
pretrained: False
batch_size: 32
num_workers: 4
max_epochs: 20
seed: 2023
lr: 0.01
optim_name: SGD
momentum: 0.9
num_folds: 5
early_stopping_monitor: loss/val_loss
early_stopping_mode: min
early_stopping_patience: 3
early_stopping_min_delta: 0.01
logger: wandb
checkpoint_callback: False
save_dir: ./outputs
```

### classifier

```bash
rye shell
python main.py dataset=yourdataset models=modelsname trainer=yourtrainer
```

#### multi-run
```bash
python main.py -m \
dataset=yourdataset1,yourdataset2 \
models=modelsname1,modelsname2 \
trainer=yourtrainer
```

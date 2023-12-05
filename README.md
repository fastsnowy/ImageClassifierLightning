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
```yaml
# config/dataset/default.yaml
name: dataset's name
train_path: path/to/train/dataset
test_path: path/to/test/dataset
aug_path: path/to/augmented/dataset
```
```yaml
# config/trainer/default.yaml
augment: False
pretrained: False
batch_size: 32
num_workers: 8
num_class: 4
max_epochs: 20
seed: 2023
lr: 0.01
optim_name: SGD
momentum: 0.9
save_dir: path/to/save_dir
```

### classifier

```bash
rye shell
python main.py dataset=yourdataset, models=modelsFileName,trainer=yourtrainer
```

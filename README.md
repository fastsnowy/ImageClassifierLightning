# imageclassifierlightning

## Setup
## prerequisites
- CUDA11.7.1
### Install dependencies
Need to install Rye
- [Rye](https://rye-up.com/guide/installation/)

### Install the package
```bash
rye sync
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
cp config/dataset/default.yaml config/dataset/yourdatasetFileName.yaml
cp config/trainer/default.yaml config/trainer/trainerFileName.yaml
```
```yaml
# config/dataset/yourdatasetFileName.yaml
name: datasetname
path:
  train: /path/to/train
  test: /path/to/test
aug:
  path: /path/to/aug
```
```yaml
# config/trainer/trainerFileName.yaml
augment: True
num_class: 4 # change your number of class
max_epochs: 20
seed: 2023
save_dir: /path/to/save # change your save path
```

### classifier

```bash
rye shell
python src/main.py dataset=yourdatasetFileName, models=modelsFileName,trainer=trainerFileName
```

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
### dataset
copy default.yaml and rename it and edit paths.
```bash
cp config/dataset/default.yaml config/dataset/yourdatasetFileName.yaml
```

### classifier
```bash
rye shell
python src/main.py dataset=yourdatasetFileName, models=modelsFileName,trainer=trainerFileName
```

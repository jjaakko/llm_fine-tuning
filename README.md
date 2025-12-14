This project contains code to to fine-tune [Mistral V3 instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) for expense categorization for accounting purposes. The repository does not contain actual data used in the thesis the project relates to, but illustrates data structures the fine-tuning code expects.

## Prerequisites

- Ensure a linux compute with `NVIDIA A100 80GB PCIe` GPU or equivalent
- Install conda (tested with version 24.5.0)

## Installing dependencies

- Clone this repository
- Install python 3.11 and Cuda by running `conda env create -f environment.yaml`
- Run `chmod u+x activate.sh`
- Activate conda environment by running `$(./activate.sh)`
- Install pinned python dependencies by running `pip install requirements/requirements.txt`
- Install the project as editable package `pip install -e .`
- Install a wrapper for tracking experiments via mlflow `pip install -e libs/tracking`

- Copy config.yaml.example to config.yaml and set data folder and fodler contains an `.env` file (`.env` file can be located within the project or outside of it)
- Copy .env.example to `.env` located in location set in `config.yaml` and change secrets accordingly (needed if measuring performance against other models than the fine-tuned model)
- Edit config.py if you want to track the experiments with mlflow in Azure Machine Learning

## Datasets for experiments

Illustrative examples of training data can be found in `sample_data/sample.jsonl`. Datatypes of each column are described in `sample_data/features.json`.

The training script expects `train`, `dev` and `test` datasets in parquet format. One can convert `sample.jsonl` to a parquet file for example as follows:

```python
from datasets import Dataset
from datasets import Features

import json

with open("features.json") as f:
    features = Features.from_dict(json.load(f))

dataset = Dataset.from_json("sample.jsonl", features=features)
dataset.to_parquet(save_path / ("train_.parquet"))
```

Illustrative example of a data structure for presenting Chart of Accounts can be found in `sample_data/coa_to_json.example.json`. 

Arrange the training data in the data folder specified in config.yaml as follows:  

```
finetuning
└── hf_dataset
    ├── coa_to_json.json
    ├── dev.parquet
    ├── test.parquet
    └── train.parquet
```

## Running the training

Run `python src/finetune/main.py`. Logs, PEFT adapter, evaluation artifacts and energy consumption information will be saved in appropriate sub folders of your data folder.

A script to test inferencing with the model after the training with a small batch is also provided in `scrips/test_inference.py`.



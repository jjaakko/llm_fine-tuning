"""Utility functions for loading and preparing datasets."""

import json
import os

import datasets
from datasets import DatasetDict, load_dataset, load_dataset_builder

import finetune.config as config
from finetune.prompt_templates import BaseTemplate


def load_hf_dataset(dataset_dir, limit: int | None = None):
    """Loads hf dataset from disk"""
    dataset_dict = datasets.load_from_disk(str(dataset_dir))
    if limit not in (0, None):
        dataset_dict = DatasetDict(
            {
                split: dataset.select(range(limit))
                for split, dataset in dataset_dict.items()
            }
        )
    else:
        dataset_dict = dataset_dict

    return dataset_dict


def load_dataset_dict_as_parquet(config):
    data_files = {
        "train": [str(config.hf_dataset_local_path / "train.parquet")],
        "dev": [str(config.hf_dataset_local_path / "dev.parquet")],
        "test": [str(config.hf_dataset_local_path / "test.parquet")],
    }
    dataset_dict = load_dataset("parquet", data_files=data_files)
    return dataset_dict


def save_dataset_dict_as_parquet(dataset_dict, save_path):
    """Saves HF Datasetdict locally to parquet file."""
    for dataset_split, dataset in dataset_dict.items():
        dataset.to_parquet(save_path / (dataset_split + ".parquet"))


def save_as_hf_dataset(hf_dataset, version, config, account_name: str):
    """Saves HF dataset in parquet format to cloud."""
    save_dataset_dict_as_parquet(hf_dataset, config.hf_dataset_local_path)

    data_files = {
        "train": [str(config.hf_dataset_local_path / "train.parquet")],
        "dev": [str(config.hf_dataset_local_path / "dev.parquet")],
        "test": [str(config.hf_dataset_local_path / "test.parquet")],
    }
    builder = load_dataset_builder(
        str(config.hf_dataset_local_path),
        data_files=data_files,
        version=version,
    )
    storage_options = {
        "account_name": account_name,
        "account_key": os.environ["STORAGE_ACCOUNT_KEY"],
    }
    output_dir = f"abfs://llm-finetune/hf_dataset/{version}"
    builder.download_and_prepare(
        output_dir,
        storage_options=storage_options,
        file_format="parquet",
    )


def load_json_file(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def prepare_dataset(
    dataset_dict: DatasetDict,
    datasets_to_map: list[str],
    prompt_template: BaseTemplate,
) -> DatasetDict:
    """Populate prompts and targets for the requested dataset splits.

    Args:
        dataset_dict: Hugging Face dataset splits to enrich with prompt fields.
        datasets_to_map: Dataset split names that require prompt generation.
        prompt_template: Prompt strategy used to build prompt strings.

    Returns:
        DatasetDict: Dataset dictionary containing generated prompts and targets for the splits.
    """
    dataset_dict = dataset_dict.map(
        lambda item: {"target": item["account_code"]}, load_from_cache_file=False
    )
    coa_to_bodyid_file = config.hf_dataset_local_path / "coa_to_json.json"
    coa_to_bodyid = load_json_file(coa_to_bodyid_file)

    for dataset in datasets_to_map:
        dataset_dict[dataset] = dataset_dict[dataset].map(
            prompt_template.get_prompt,
            fn_kwargs={"coa_to_bodyid": coa_to_bodyid[dataset]},
            load_from_cache_file=False,
        )
    return dataset_dict

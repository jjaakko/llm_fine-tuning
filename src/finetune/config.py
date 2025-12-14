from pathlib import Path
from typing import Literal

import pyprojroot
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class Config(BaseModel):
    # Model to do inference with when evaluating performance.
    llm_type_for_perfomance_measurement: Literal[
        "azure-openai",
        "mistral7B-instruct",
        "fake",
        "finetuned-mistral7B-instruct",
    ] = "finetuned-mistral7B-instruct"
    # Whether to use mlflow for tracking the experiments.
    tracker_type: Literal["mlflow_tracker_azure", "no_tracker"] = "no_tracker"
    # Name of the experiment to track in mlflow.
    experiment_name: str = "[REPLACE WITH EXPERIMENT NAME]"
    # Separate experiments with different datasets with version. Can be arbitrary.
    version: str = "1.0"


config = Config()


def load_config():
    config_file_name = "config.yaml"
    try:
        config_file_path = get_project_root() / str(Path(config_file_name))
        with open(config_file_path) as f:
            config_from_file = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {config_file_path} not found.")
    return config_from_file


def get_project_root():
    base_dir = pyprojroot.find_root(pyprojroot.has_file("pyproject.toml"))
    return base_dir


def get_secrets_path(environment):
    config = load_config()
    secrets_path = Path(config["secrets_path"].format(environment=environment))
    return secrets_path


def get_data_path(environment: str, version: str) -> Path:
    config = load_config()
    data_path = Path(config["data_path"].format(environment=environment)) / version
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    return data_path


environment = "stage"
print(f"ACTIVE ENVIRONMENT: {environment}")

project_root = get_project_root()
secrets_path = get_secrets_path(environment)
version = config.version
data_path = get_data_path(environment, version)
hf_dataset_local_path = data_path / "finetuning" / "hf_dataset"
hf_dataset_local_path.mkdir(exist_ok=True, parents=True)

datasets = ["train", "dev", "test"]

max_token_length_before_applying_chat_template = 864

load_dotenv(dotenv_path=secrets_path)

print(f"Using data path: {data_path}")
print(f"Using secrets base path: {secrets_path}")

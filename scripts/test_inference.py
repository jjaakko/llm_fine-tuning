import time

from transformers import (
    AutoTokenizer,
)

import finetune.config as config
import finetune.data_utils as data_utils
import finetune.finetune_utils as finetune_utils
from finetune.prompt_templates import (
    OneStepSubCategoryClassification,
)

### LOAD DATA ###

dataset_dict = data_utils.load_dataset_dict_as_parquet(config)
dataset_dict = dataset_dict.filter(lambda example, index: index < 10, with_indices=True)

dataset_dict = data_utils.prepare_dataset(
    dataset_dict,
    config.datasets,
    OneStepSubCategoryClassification(),
)

### LOAD MODEL AND PEFT ADAPTER ###
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    padding="max_length",
    truncation=True,
    max_length=config.max_token_length_before_applying_chat_template,
)
tokenizer.pad_token = tokenizer.eos_token

output_dir = config.data_path / "models"
peft_model_path = output_dir / "peft_model"

llm = finetune_utils.get_mistral_instruct(
    peft_model_path=peft_model_path, is_trainable=False
)

### INFERENCE WITH A SMALL BATCH ###

prompts = [dataset_dict["test"][index]["prompt"] for index in range(0, 2)]

print("Starting inference")
start = time.time()
result = llm.batch(prompts)
end = time.time()
print(result)
print(f"Inference for {len(prompts)} items took {end - start} seconds")

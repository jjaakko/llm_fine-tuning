import os
from pathlib import Path

import torch
from langchain_openai import AzureChatOpenAI
from peft import PeftModel
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from finetune.fake_chat_model import FakeChatModel
from finetune.MistralLLM import MistralLLM


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024**2} MB.")


def get_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    device = get_device()
    model.to(device)
    return model


def get_quantized_model(model_name):
    compute_dtype = getattr(torch, "float16")
    # compute_dtype = getattr(torch, "bfloat16")  # Or torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    # device_map = {"": 0}
    device_map = (
        {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_auth_token=True,
        use_cache=True,
    )
    return model


def get_mistral_instruct(is_trainable: bool, peft_model_path: Path | None = None):
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        padding="max_length",
        truncation=True,
        max_length=640,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = get_quantized_model(model_name)
    if peft_model_path:
        print(f"Loaded model with peft model path: {str(peft_model_path)}")

    # Inject fine-tuned adapter.
    if peft_model_path:
        model = PeftModel.from_pretrained(
            model,
            str(peft_model_path),
            # torch_dtype=torch.bfloat16,
            is_trainable=is_trainable,
        )
    llm = MistralLLM(model=model, tokenizer=tokenizer)
    return llm


def tokenize_function(instance, tokenizer, max_length):
    tokenization = tokenizer(
        instance["prompt"],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    instance["input_ids"] = tokenization.input_ids
    instance["attention_mask"] = tokenization.attention_mask

    instance["labels"] = tokenizer(
        list(map(str, instance["account_code"])),
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    return instance


def compute_length(instance, index, tokenizer, column):
    instance["length"] = len(
        tokenizer(instance[column], truncation=False, padding=False).input_ids
    )
    instance["index"] = index
    return instance


def filter_based_on_length(dataset, tokenizer, max_length, column):
    dataset = dataset.map(
        compute_length,
        with_indices=True,
        fn_kwargs={"tokenizer": tokenizer, "column": column},
    )
    dataset = dataset.filter(lambda instance: instance["length"] <= max_length)

    return dataset


def get_llm(llm_type: str, model_path=None):
    if llm_type == "fake":
        llm = FakeChatModel()
    elif llm_type == "azure-openai":
        azure_chat_open_ai_params = {
            "azure_endpoint": os.getenv("OPENAI_AZURE_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
            "deployment_name": os.getenv("DEPLOYMENT_NAME"),
            "openai_api_key": os.getenv("OPENAI_API_KEY_AZURE"),
            "temperature": 0,
            "max_tokens": 1000,
            "request_timeout": 20,
        }
        llm = AzureChatOpenAI(**azure_chat_open_ai_params)
    elif llm_type == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model="mistral-nemo",
            temperature=0,
        )
    elif llm_type == "finetuned-mistral7B-instruct":
        import finetune.config as config

        output_dir = config.data_path / "models"
        peft_model_path = model_path if model_path else output_dir / "peft_model"
        llm = get_mistral_instruct(peft_model_path=peft_model_path, is_trainable=False)
    elif llm_type == "mistral7B-instruct":
        llm = get_mistral_instruct(peft_model_path=None, is_trainable=False)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    return llm

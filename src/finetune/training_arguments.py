"""Training arguments and configurations related tightly rto the training runs"""
from finetune.prompt_templates import (
    OneStepSubCategoryClassification,
    OneStepSubCategoryClassificationCreateReasoningStep,
    OneStepSubCategoryClassificationUtilizeReasoningStep,
)


def get_default_training_args(output_dir):
    training_args = {
        "fp16": True,
        "do_eval": True,
        # "evaluation_strategy": "epoch",
        # "gradient_accumulation_steps": 128,
        # "gradient_checkpointing": True,
        # "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # "lr_scheduler_type": "cosine",
        "overwrite_output_dir": True,
        "seed": 42,
        "per_device_train_batch_size": 20,
        # "logging_dir": "./logs",
        # "logging_steps": 500,
        "logging_steps": 50,
        # "https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.log_level"
        "log_level": "info",
        "logging_first_step": True,
        # "auto_find_batch_size": True,
        # "per_gpu_train_batch_size": 16,
        "num_train_epochs": 1,
        # "max_steps": -1,
        # "max_steps": 2,
        # "https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.load_best_model_at_end"
        "load_best_model_at_end": True,
        # "https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.save_strategy"
        "save_strategy": "steps",
        # "save_strategy": "no",
        # "https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.save_steps"
        "save_steps": 900,
        "evaluation_strategy": "steps",
        "eval_steps": 60,
        "optim": "paged_adamw_8bit",
        "eval_strategy": "steps",
        "learning_rate": 8.121989995384603e-05,
        "weight_decay": 0.001,
        "warmup_ratio": 0.3,
        "group_by_length": True,
        "lr_scheduler_type": "linear",
        "report_to": "none",
        # "report_to": "mlflow",
        "output_dir": output_dir,
    }
    return training_args


def get_hyperparam_training_args(trial):
    args = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-4),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [20, 40]
        ),
    }
    return args


def get_default_lora_args():
    # {'r': 16, 'lora_alpha': 124, 'learning_rate': 1.73279951379735e-05, 'per_device_train_batch_size': 40}
    args = {
        "r": 64,
        "lora_alpha": 32,
        # "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "target_modules": "all-linear",
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    return args


def get_hyperparam_lora_args(trial):
    args = {
        "r": trial.suggest_categorical("r", [16, 32, 64]),
        "lora_alpha": trial.suggest_categorical("lora_alpha", [16, 32, 64, 124]),
    }
    return args


# strategy = "use_reasoning_step"
strategy = "predict_code_without_reasoning"
if strategy == "use_reasoning_step":
    # Contains target (the correct accountcode).
    prompt_template_during_training = (
        OneStepSubCategoryClassificationCreateReasoningStep()
    )

    # Does not contain target.
    prompt_template_during_inference = (
        OneStepSubCategoryClassificationUtilizeReasoningStep()
    )
elif strategy == "predict_code_without_reasoning":
    prompt_template_during_training = OneStepSubCategoryClassification()
    prompt_template_during_inference = OneStepSubCategoryClassification()
else:
    raise ValueError("Not valid strategy.")


def get_other_args():
    actual_run = True
    args = {
        # Limiting data set size during training for debugging purposes. If zero, then use all samples.
        "limit_for_train_data": 0 if actual_run else 100,
        "limit_for_dev_data": 100 if actual_run else 2,
        "padding_side_during_training": "right",
        # Limit number of samples used in inference for development purposes.
        "n_inference_samples": 0 if actual_run else 2,
        "chunk_size": 25 if actual_run else 2,
        "n_trials": 20 if actual_run else 2,
        # When fixed_run is true, we run a single run without hyperparameter tuning.
        "fixed_run": True,
        "evaluate_at": [0.10, 0.15, 0.20, 0.25, 0.5, 0.75] if actual_run else [],
        "debugging_run": not actual_run,
        "strategy": strategy,
        # "prompt_strategy": prompt_strategy.__class__.__name__
        "prompt_template_during_training": prompt_template_during_training.__class__.__name__,
        "prompt_template_during_inference": prompt_template_during_inference.__class__.__name__,
    }
    return args

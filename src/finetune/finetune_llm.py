import logging
import os
import sys

import mlflow
import optuna
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

import finetune.config as config
import finetune.data_utils as data_utils
import finetune.finetune_utils as finetune_utils
import finetune.performance as performance
from finetune.training_arguments import (
    get_default_lora_args,
    get_default_training_args,
    get_hyperparam_lora_args,
    get_hyperparam_training_args,
    get_other_args,
)
from ml_tracking.ml_tracker import ML_Tracker

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Keep track of the best accuracy if doing hyperparameter optimization with optuna.
best_accuracy = 0.0


class FractionEvaluationCallback(TrainerCallback):
    """Save mode and measure performance at given fractions"""

    def __init__(self, evaluate_at, eval_dataset, tracker: ML_Tracker, logger):
        """
        Args:
            evaluate_at (list of float): fractions (between 0 and 1) at which to save/evaluate.
            eval_dataset: the dataset to run evaluation on.
        """
        self.evaluate_at = sorted(evaluate_at)
        self.eval_dataset = eval_dataset
        self.evaluated = set()
        self.tracker = tracker
        self.logger = logger

    def on_step_begin(self, args, state, control, **kwargs):
        current_fraction = state.global_step / state.max_steps
        for frac in self.evaluate_at:
            if frac not in self.evaluated and current_fraction >= frac:
                self.evaluated.add(frac)
                model = kwargs.get("model", None)

                # Define a subdirectory to save the checkpoint corresponding to this fraction.
                save_dir = os.path.join(
                    args.output_dir, f"checkpoint_fraction_{int(frac * 100)}"
                )
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                print(
                    f"Saved checkpoint at {int(frac * 100)}% progress (step {state.global_step}/{state.max_steps}) to {save_dir}"
                )
                # Evaluate on the provided eval_dataset
                with self.tracker.start_run(nested=True):
                    self.tracker.log_params(
                        {
                            "global_step": state.global_step,
                            "frac": frac,
                            "model_dir": save_dir,
                        }
                    )
                    dev_metrics = performance.performance_run(
                        dataset_type="dev",
                        tracker=self.tracker,
                        logger=self.logger,
                        model_path=save_dir,
                    )
                    self.tracker.log_metrics(dev_metrics)
                    test_metrics = performance.performance_run(
                        dataset_type="test",
                        tracker=self.tracker,
                        logger=self.logger,
                        model_path=save_dir,
                    )
                    self.tracker.log_metrics(test_metrics)
                    self.tracker.log_artifact(str(save_dir), ".")

                continue
        return control


def apply_chat_template(instance, tokenizer, target_column: str):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": instance["prompt"]},
        {"role": "assistant", "content": instance[target_column]},
    ]
    # Tokenization is handled internally.
    # SFTTrainer gets max_seq_length as an argument.
    instance["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return instance


def configure_transformer_logging(transformers_log_file):
    transformers.logging.set_verbosity_info()

    # Set up a file handler for transformers logging
    file_handler = logging.FileHandler(transformers_log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    # Define a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Get the transformers logger and add the file handler to it
    transformers_logger = transformers.logging.get_logger()
    transformers_logger.addHandler(file_handler)

    print(transformers.utils.logging.get_verbosity())


def objective(
    trial,
    logger,
    model_name,
    train_set,
    eval_set,
    tokenizer,
    tracker,
    best_model_path,
    nested,
):
    """Objective function required in hyperparameter tuning."""
    if nested:
        tracker.start_run_(nested=True)

    run = mlflow.last_active_run()
    if run:
        logger.info(f"Objective function for {run.info.run_name} called")

    ### GET MODEL ###
    original_model = finetune_utils.get_quantized_model(model_name)
    logger.debug("Model loaded")

    ### PEFT TRAINING SETUP ###
    original_model = prepare_model_for_kbit_training(original_model)

    lora_config = {**get_default_lora_args(), **get_hyperparam_lora_args(trial)}
    peft_config = LoraConfig(**lora_config)

    peft_model = get_peft_model(original_model, peft_config)
    print()
    print(finetune_utils.print_number_of_trainable_model_parameters(peft_model))

    ### TRAINING ###
    output_dir = config.data_path / "models"
    peft_model_path = output_dir / "peft_model"
    peft_model_path.mkdir(exist_ok=True, parents=True)

    training_args = {
        **get_default_training_args(output_dir=output_dir),
        **get_hyperparam_training_args(trial),
    }
    tracker.log_params(training_args)
    peft_training_args = TrainingArguments(**training_args)

    peft_trainer = SFTTrainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1600,
    )

    # If fixed_run is true, attach the callback to evaluate at given fractions.
    other_args = get_other_args()
    if other_args.get("fixed_run", False):
        fraction_callback = FractionEvaluationCallback(
            evaluate_at=other_args.get("evaluate_at", []),
            eval_dataset=eval_set,
            tracker=tracker,
            logger=logger,
        )
        # Add the callback to the trainer.
        peft_trainer.add_callback(fraction_callback)

    logger.info("Starting peft training")
    peft_trainer.train()

    peft_trainer.model.save_pretrained(output_dir / peft_model_path)
    tracker.log_params(trial.params)

    logger.info("Trial params:")
    logger.info(trial.params)
    logger.info("Start performance script")
    metrics = performance.performance_run(
        dataset_type="dev", tracker=tracker, logger=logger
    )
    tracker.log_metrics(metrics)
    accuracy = metrics["dev_accuracy"]
    logger.info(f"Performance script completed with acc={accuracy}")

    global best_accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        peft_trainer.model.save_pretrained(best_model_path)

    if nested:
        tracker.end_run()

    return accuracy


def main(transformers_log_file, tracker: ML_Tracker, logger):
    configure_transformer_logging(transformers_log_file)
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    device = finetune_utils.get_device()
    logger.info(f"device: {device}")

    ### LOAD AND PREPARE DATASET ###
    dataset_dict = data_utils.load_dataset_dict_as_parquet(config)

    args = get_other_args()

    # Speed up training by using smaller dev dataset if limit has been set.
    if args["limit_for_dev_data"] > 0:
        dataset_dict["dev"] = dataset_dict["dev"].filter(
            lambda example, index: index < args["limit_for_dev_data"], with_indices=True
        )

    if args["limit_for_train_data"] > 0:
        dataset_dict["train"] = dataset_dict["train"].filter(
            lambda example, index: index < args["limit_for_train_data"],
            with_indices=True,
        )

    from finetune.training_arguments import prompt_template_during_training

    decoded_dataset_dict = data_utils.prepare_dataset(
        dataset_dict,
        config.datasets,
        prompt_template=prompt_template_during_training,
    )

    ### TOKENIZER & PREPROCESSING

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side=args["padding_side_during_training"]
    )
    tokenizer.pad_token = tokenizer.eos_token

    decoded_dataset_dict = decoded_dataset_dict.map(
        finetune_utils.compute_length,
        with_indices=True,
        fn_kwargs={"tokenizer": tokenizer, "column": "prompt"},
    )
    decoded_dataset_dict = finetune_utils.filter_based_on_length(
        decoded_dataset_dict,
        tokenizer,
        max_length=config.max_token_length_before_applying_chat_template,
        column="prompt",
    )
    args["max_token_length_before_applying_chat_template"] = (
        config.max_token_length_before_applying_chat_template
    )

    ### APPLY CHAT TEMPLATE AND REMOVE OBSOLETE COLUMNS ###
    column_names = list(decoded_dataset_dict["train"].features)
    decoded_dataset_dict = decoded_dataset_dict.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "target_column": prompt_template_during_training.target_column,
        },
        remove_columns=column_names,
        desc="Applying chat template",
    )

    output_dir = config.data_path / "models"
    peft_model_path = output_dir / "peft_model"
    best_model_path = output_dir / "best_peft_model"
    import shutil

    if peft_model_path.exists():
        shutil.rmtree(peft_model_path)
    if best_model_path.exists():
        shutil.rmtree(best_model_path)
    peft_model_path.mkdir(exist_ok=True, parents=True)
    best_model_path.mkdir(exist_ok=True)

    def single_run():
        default_args = {
            **{},
            **get_default_lora_args(),
            **get_default_training_args(output_dir=output_dir),
        }
        objective(
            optuna.trial.FixedTrial(default_args),
            logger=logger,
            model_name=model_name,
            train_set=decoded_dataset_dict["train"],
            eval_set=decoded_dataset_dict["dev"],
            tokenizer=tokenizer,
            tracker=tracker,
            best_model_path=str(best_model_path),
            nested=False,
        )
        tracker.log_artifact(str(peft_model_path), ".")

    def multiple_runs():
        from datetime import datetime

        from mlflow.utils.name_utils import _generate_random_name

        study_name = (
            _generate_random_name() + "--" + datetime.now().strftime("%d-%b--%H-%M")
        )
        trials_path = config.data_path / "trials"
        trials_path.mkdir(exist_ok=True)
        storage_name = f"sqlite:///{str(trials_path)}/{study_name}.db"
        args["storage_name"] = storage_name
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: objective(
                trial,
                logger=logger,
                model_name=model_name,
                train_set=decoded_dataset_dict["train"],
                eval_set=decoded_dataset_dict["dev"],
                tokenizer=tokenizer,
                tracker=tracker,
                best_model_path=best_model_path,
                nested=True,
            ),
            n_trials=args["n_trials"],
        )
        best_trial = study.best_trial
        tracker.log_metric("accuracy_on_dev_set", best_trial.value)
        tracker.log_params(study.best_params)
        tracker.log_artifact(str(best_model_path), ".")

        return study.best_params

    args["model_name"] = model_name
    tracker.log_params(args)

    if args["fixed_run"]:
        single_run()
    elif not args["fixed_run"]:
        multiple_runs()
    else:
        raise ValueError("fixed_run argument is required")


if __name__ == "__main__":
    from huggingface_hub import login

    access_token = os.environ["ACCESS_TOKEN_HF"]
    login(token=access_token)

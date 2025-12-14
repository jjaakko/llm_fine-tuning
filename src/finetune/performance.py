"""Model evaluation"""
import json
from time import time

import numpy as np
import pandas as pd
from datasets import DatasetDict
from dotenv import load_dotenv
from langchain_community.callbacks.manager import get_openai_callback
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI

import finetune.config as config
import finetune.data_utils as data_utils
import finetune.finetune_utils as finetune_utils
import finetune.logging_config as logging_config
from finetune import coa
from finetune.MistralLLM import MistralLLM
from finetune.training_arguments import get_other_args, prompt_template_during_inference
from ml_tracking.ml_tracker import ML_Tracker

load_dotenv(dotenv_path=config.secrets_path)


def get_prompts_and_targets(dataset_type):
    logger = logging_config.get_logger(__file__, "INFO")
    folder = config.data_path / "finetuning"
    filename = folder / f"{dataset_type}_set_{config.bodyids[0]}.json"

    with open(filename) as f:
        test_set = json.load(f)

    logger.info(f"Prompt {dataset_type} dataset full size:{len(test_set)}")

    return test_set


def predict_dataset(
    llm: MistralLLM | ChatOllama | AzureChatOpenAI,
    dataset: DatasetDict,
    dataset_type: str,
    limit: int | None = None,
    batch: bool = True,
    chunk_size: int = 10,
):
    """Request and collect predictions for the given dataset.

    Args:
        llm (_type_): _description_
        dataset (_type_): _description_
        dataset_type (str): _description_
        limit (int | None, optional): _description_. Defaults to None.
        batch (bool, optional): _description_. Defaults to True.
        chunk_size (int, optional): _description_. Defaults to 10.

    Raises:
        a: _description_
        RateLimitException: _description_

    Returns:
        _type_: _description_
    """
    logger = logging_config.get_logger(__file__, "INFO")
    usage_metadatas = []
    log_every_nth = 1
    if limit in (0, None):
        limit = len(dataset)

    logger.info("Starting to prompt LLM")
    pred_y = []

    if llm._llm_type == "custom":
        for i in range(0, limit, chunk_size):
            chunk = dataset[dataset_type]["prompt"][i : i + chunk_size]
            pred_y.extend(llm.batch(chunk))
            logger.info(
                f"Processed chunk {i // chunk_size + 1}/{(limit + chunk_size - 1) // chunk_size}"
            )
    else:
        import concurrent.futures
        import re

        import openai
        from tenacity import retry, retry_if_exception_type, stop_after_attempt

        # Define a custom exception for rate limiting.
        class RateLimitException(Exception):
            def __init__(self, wait_time, message="Rate limit hit"):
                self.wait_time = wait_time
                super().__init__(message)

        def extract_wait_time(message):
            """
            Extract the wait time in seconds from the API error message.
            For example: "Please retry after 30 seconds" returns 30.
            """
            match = re.search(r"Please retry after (\d+) seconds", message)
            if match and match.group(1):
                return int(match.group(1))
            return 20

        def custom_wait(retry_state):
            """
            Custom wait function for tenacity. Uses the wait_time from a RateLimitException if present.
            """
            exc = retry_state.outcome.exception()
            if isinstance(exc, RateLimitException):
                return exc.wait_time
            return 5  # default wait time for other exceptions

        # Wrap the API call with tenacity retry logic.
        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type((RateLimitException, Exception)),
            wait=custom_wait,
        )
        def safe_invoke(prompt):
            """
            Call the API via llm.invoke, retrying up to 3 times if exceptions occur.
            If the API returns an error message indicating rate limiting,
            raise a RateLimitException with an appropriate wait time.
            """
            try:
                result = llm.invoke(prompt)
            except openai.RateLimitError as e:
                # If you can inspect the result to detect rate limiting, do so:
                # For example, if result is a dict with an error:
                print(e)
                wait_time = extract_wait_time(str(e))
                raise RateLimitException(wait_time, str(e))
            except Exception as e:
                print(e)
                raise
            return result

        import signal
        import sys

        interrupted = False

        checkpoint_file = (
            config.data_path / "performance" / "openai_prompting_checkpoint.json"
        )

        # Set up signal handler
        def signal_handler(sig, frame):
            nonlocal interrupted
            if interrupted:  # If pressed twice, exit immediately
                logger.warning("Forced exit. Some data may be lost.")
                sys.exit(1)
            logger.warning(
                "KeyboardInterrupt detected. Saving results and exiting gracefully..."
            )
            interrupted = True
            # Let the execution continue to save the results

        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)
        with get_openai_callback() as cb:
            import math

            upper_bound = math.ceil(limit / chunk_size)
            for chunk in range(0, upper_bound, 1):
                if interrupted:
                    print("Interrupted")
                    break

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=chunk_size
                ) as executor:
                    # Submit all API calls concurrently with robust retry handling.
                    futures = [
                        executor.submit(safe_invoke, dataset[dataset_type][i]["prompt"])
                        for i in range(
                            chunk * chunk_size, min((chunk + 1) * chunk_size, limit)
                        )
                    ]
                    print("futures send")
                    for i, future in enumerate(futures):
                        try:
                            result = future.result()
                            result = result.content
                        except Exception as e:
                            logger.error(f"Error processing prompt at index {i}: {e}")
                            result = None  # Optionally handle failures differently.
                        usage_metadatas.append(0)
                        pred_y.append(result)
                        if (i + 1) % log_every_nth == 0 or (i + 1) == limit:
                            logger.info(
                                f"Generated {chunk * chunk_size + i + 1}/{limit} prediction"
                            )

                with open(checkpoint_file, "w") as f:
                    json.dump(pred_y, f)
                logger.info(f"Checkpoint saved with {len(pred_y)} results")
            print(cb)

    return pred_y


def evaluate_dataset(pred_y, test_y):
    pred_y = np.array(pred_y)
    test_y = np.array(test_y)
    accuracy = sum(pred_y == test_y) / len(test_y)
    return accuracy


def add_num_parents(instance, full_coa):
    # instance["account_code"] = 1730
    i, parents = coa.get_parents(
        full_coa[instance["full_coa"]],
        instance["account_code"],
        full_element=True,
        include_account_name=True,
    )
    forced_indexes = [parent["index"] for parent in parents]
    categories = coa.get_consolidated_elements_from_root_by_data(
        [full_coa[instance["full_coa"]][i]],
        indexes=[0],
        level_limit=100,
        forced_indexes=forced_indexes,
        name_key="name_en",
    )

    categories_str_, number_of_targets = coa.get_coa_as_string_and_leaf_count(
        categories, name_key="name_en"
    )
    instance["num_targets"] = number_of_targets
    return instance


def get_eval_comparison_df(
    llm_type: str, dataset_type: str, limit: int | None = None, model_path=None
):
    logger = logging_config.get_logger(__file__, "INFO")
    llm = finetune_utils.get_llm(llm_type=llm_type, model_path=model_path)

    finetuning_dataset_dict = data_utils.load_dataset_dict_as_parquet(config)

    if limit in (0, None):
        limit = len(finetuning_dataset_dict[dataset_type])

    # If the dataset size is not filtered, it's slow to load the whole dataset because of coa gets duplicated.
    finetuning_dataset_dict = finetuning_dataset_dict.filter(
        lambda example, index: index < limit, with_indices=True
    )

    args = get_other_args()
    finetuning_dataset_dict = data_utils.prepare_dataset(
        finetuning_dataset_dict,
        datasets_to_map=[dataset_type],
        prompt_template=prompt_template_during_inference,
    )

    start_time_in_seconds = time()
    chunk_size = args["chunk_size"]
    pred_y = predict_dataset(
        llm,
        finetuning_dataset_dict,
        dataset_type=dataset_type,
        limit=limit,
        batch=True,
        chunk_size=chunk_size,
    )
    end_time_in_seconds = time()
    throughput = len(finetuning_dataset_dict[dataset_type]) / (
        end_time_in_seconds - start_time_in_seconds
    )
    logger.info(
        f"Inference for {limit} items took {end_time_in_seconds - start_time_in_seconds} seconds"
    )
    logger.info(f"Throughput: {throughput:.2f} prompts per second")
    test_y = finetuning_dataset_dict[dataset_type]["account_code"]

    df = pd.DataFrame(
        {
            "receiver_id": finetuning_dataset_dict[dataset_type]["receiver_id"],
            "product": finetuning_dataset_dict[dataset_type]["product_name"],
            "supplier_business": finetuning_dataset_dict[dataset_type][
                "invoice_sender_main_business_line"
            ],
            "pred_y": pred_y,
            "test_y": list(map(str, test_y)),
            "prompt": finetuning_dataset_dict[dataset_type]["prompt"],
            "categories": finetuning_dataset_dict[dataset_type]["categories"],
            "num_targets": finetuning_dataset_dict[dataset_type]["num_targets"],
            "test_y_check": finetuning_dataset_dict[dataset_type]["account_code"],
            "price": finetuning_dataset_dict[dataset_type]["price"],
            "supplier_name": finetuning_dataset_dict[dataset_type]["invoice_sender"],
        }
    )
    from finetune.text_extraction import extract_prediction

    if llm._llm_type == "custom":
        # Exclude prompt from the completion.
        df["pred_y"] = df["pred_y"].apply(
            lambda item: item.split("[/INST]")[1].replace("</s>", "").strip()
        )

    # Extract prediction.
    df["pred_y_extracted"] = df["pred_y"].apply(lambda item: extract_prediction(item))

    return df


def performance_run(dataset_type, tracker: ML_Tracker, logger, model_path=None):
    args = get_other_args()
    tracker.log_params({"n_inference_samples": args["n_inference_samples"]})
    df = get_eval_comparison_df(
        llm_type=config.config.llm_type_for_perfomance_measurement,
        dataset_type=dataset_type,
        limit=args["n_inference_samples"],
        model_path=model_path,
    )
    results_path = config.data_path / "performance"
    results_path.mkdir(exist_ok=True)
    df.to_csv(
        config.data_path
        / "performance"
        / f"performance_data-{dataset_type}-{config.config.llm_type_for_perfomance_measurement}-{config.environment}.csv"
    )
    df.to_json(
        config.data_path
        / "performance"
        / f"performance_data-{dataset_type}-{config.config.llm_type_for_perfomance_measurement}-{config.environment}.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )
    tracker.log_artifact(
        config.data_path
        / "performance"
        / f"performance_data-{dataset_type}-{config.config.llm_type_for_perfomance_measurement}-{config.environment}.csv"
    )

    acc = evaluate_dataset(df["pred_y_extracted"], df["test_y"])
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        f"{dataset_type}_accuracy": accuracy_score(
            df["test_y"], df["pred_y_extracted"]
        ),
        f"{dataset_type}_precision_macro": precision_score(
            df["test_y"], df["pred_y_extracted"], average="macro"
        ),
        f"{dataset_type}_recall_macro": recall_score(
            df["test_y"], df["pred_y_extracted"], average="macro"
        ),
        f"{dataset_type}_f1_macro": f1_score(
            df["test_y"], df["pred_y_extracted"], average="macro"
        ),
        f"{dataset_type}_baseline_acc": df["num_targets"].apply(lambda x: 1 / x).sum()
        / df.shape[0],
    }

    logger.info(f"Model {acc=}")
    logger.info(f"{df['pred_y'].shape[0]=}")

    return metrics

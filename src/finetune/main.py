import sys

from codecarbon import OfflineEmissionsTracker

import finetune.config as config
import finetune.finetune_llm as finetune_llm
import finetune.logging_config as logging_config
import finetune.performance as performance
from ml_tracking.get_tracker import get_tracker


def main():
    (config.data_path / "logs").mkdir(exist_ok=True)
    log_file = config.data_path / "logs" / "training_output.log"

    # Redirect stdout to both terminal and file
    sys.stdout = logging_config.StreamToFile(log_file, sys.stdout)
    sys.stderr = logging_config.StreamToFile(log_file, sys.stderr)

    logger = logging_config.get_logger(__file__, "INFO")

    logger.info(f"Running finetuning with version {config.version}")

    logger.info("Start finetuning script")
    tracker = get_tracker(config.config.tracker_type)
    with tracker.start_run(config.config.experiment_name):
        with OfflineEmissionsTracker(
            country_iso_code="SWE", output_file=str(config.data_path / "emissions.csv")
        ):
            finetune_llm.main(
                config.data_path / "logs" / "transformers_training.log", tracker, logger
            )
            tracker.log_artifact(str(config.data_path / "logs"), ".")

        logger.info("Finetuning completed")
        tracker.log_artifact(str(config.data_path / "emissions.csv"), ".")

        logger.info("Evaluating test performance")
        metrics = performance.performance_run(
            dataset_type="test", tracker=tracker, logger=logger
        )
        tracker.log_metrics(metrics)

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()

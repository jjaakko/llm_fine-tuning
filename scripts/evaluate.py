from pathlib import Path

import finetune.config as config
from finetune.logging_config import get_logger
from finetune.performance import performance_run
from ml_tracking.get_tracker import get_tracker

if __name__ == "__main__":
    logger = get_logger(__file__, "INFO")
    tracker = get_tracker(config.config.tracker_type)
    with tracker.start_run(experiment_name=config.config.experiment_name):
        dataset_type = "test"
        model_path = Path("<REPLACE-ME>")

        metrics = performance_run(
            dataset_type="test", tracker=tracker, logger=logger, model_path=model_path
        )
        tracker.log_params(
            {
                "llm_type": config.config.llm_type_for_perfomance_measurement,
                "dataset_type": dataset_type,
                "dataset_version": config.version,
            }
        )
        tracker.log_metrics(metrics)

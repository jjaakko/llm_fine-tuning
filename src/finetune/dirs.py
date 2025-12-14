from datetime import datetime

import finetune.config as config

timestamp = datetime.now().strftime("%d-%b-%Y_%H_%M")
(config.data_path / "logs").mkdir(exist_ok=True)
log_file = config.data_path / "logs" / "training_output_explicit.log"

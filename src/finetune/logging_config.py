import logging

from finetune.dirs import log_file


class StreamToFile:
    def __init__(self, filename, stream):
        self.terminal = stream  # This can be sys.stdout or sys.stderr
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Common formatter.
formatter = logging.Formatter(fmt="%(levelname)s %(asctime)s %(message)s\n")

# Common handler for all loggers defined in the project (excluding other loggers)
file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
file_handler.setFormatter(fmt=formatter)
file_handler.setLevel("INFO")


def get_logger(name: str, level: str):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # Return the existing logger to avoid adding multiple handlers.
        return logger

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=formatter)
    stream_handler.setLevel(level)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger

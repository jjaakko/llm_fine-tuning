"""Interface for tracking ML jobs and experiments."""

from contextlib import contextmanager
from typing import Protocol


class ML_Tracker(Protocol):
    def __init__(self, tracking_uri):
        pass

    @contextmanager
    def start_run(self, experiment_name: str, **kwargs):
        """Each implementation of this class should have start_run to return a class with context manager."""
        raise NotImplementedError(type(self))

    def end_run(self):
        raise NotImplementedError(type(self))

    def log_metric(self, key, value):
        raise NotImplementedError(type(self))

    def log_metrics(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def log_input(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def log_params(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def log_artifact(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def log_figure(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def log_sklearn_model(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def dataset_from_pandas(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def search_runs(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def load_sklearn_model(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def get_run(self, run_id):
        raise NotImplementedError(type(self))

    def set_tags(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def autolog(self):
        raise NotImplementedError(type(self))

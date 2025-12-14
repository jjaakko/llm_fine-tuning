from contextlib import contextmanager


class NoTracker:
    def __init__(self, tracking_uri):
        pass

    @contextmanager
    def start_run(self, experiment_name=None, **kwargs):
        yield

    def log_params(self, *args, **kwargs):
        pass

    def log_metric(self, key, value):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def log_figure(self, *args, **kwargs):
        pass

    def log_input(self, *args, **kwargs):
        pass

    def dataset_from_pandas(self, *args, **kwargs):
        pass

    def log_sklearn_model(self, *args, **kwargs):
        pass

    def search_runs(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def load_sklearn_model(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def get_run(self, run_id):
        raise NotImplementedError(type(self))

    def end_run(self):
        pass

    def set_tags(self, *args, **kwargs):
        pass

    def autolog(self):
        raise NotImplementedError(type(self))

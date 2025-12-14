from contextlib import contextmanager

import mlflow
import mlflow.data
import mlflow.exceptions
import mlflow.sklearn


class MlflowTracker:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)

    @contextmanager
    def start_run(self, *args, **kwargs):
        """Start run and as a convenience set experiment if provided."""
        # See: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
        experiment_name = kwargs.get("experiment_name", None)
        kwargs.pop("experiment_name", None)
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run(*args, **kwargs) as run:
            yield run

    def start_run_(self, *args, **kwargs):
        """Start run and as a convenience set experiment if provided."""
        # See: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
        experiment_name = kwargs.get("experiment_name", None)
        kwargs.pop("experiment_name", None)
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

        mlflow.start_run(*args, **kwargs)

    def end_run(self):
        mlflow.end_run()

    def log_params(self, *args, **kwargs):
        # See: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_params
        mlflow.log_params(*args, **kwargs)

    def log_metric(self, key, value):
        mlflow.log_metric(key=key, value=value)

    def log_metrics(self, *args, **kwargs):
        mlflow.log_metrics(*args, **kwargs)

    def log_sklearn_model(self, *args, **kwargs):
        # See: https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model
        mlflow.sklearn.log_model(*args, **kwargs)

    def log_input(self, *args, **kwargs):
        # See: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_input
        mlflow.log_input(*args, **kwargs)

    def log_artifact(self, *args, **kwargs):
        # See: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifact
        mlflow.log_artifact(*args, **kwargs)

    def log_figure(self, *args, **kwargs):
        # See: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_figure
        mlflow.log_figure(*args, **kwargs)

    def dataset_from_pandas(self, *args, **kwargs):
        # See: https://mlflow.org/docs/latest/python_api/mlflow.data.html#mlflow.data.from_pandas
        return mlflow.data.from_pandas(*args, **kwargs)

    def search_runs(self, *args, **kwargs):
        # See: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs
        return mlflow.search_runs(*args, **kwargs)

    def load_sklearn_model(self, *args, **kwargs):
        # See: https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.load_model
        return mlflow.sklearn.load_model(*args, **kwargs)

    def get_run(self, run_id: str):
        # See: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.get_run
        return mlflow.get_run(run_id)

    def set_tags(self, *args, **kwargs):
        return mlflow.set_tags(*args, **kwargs)

    def autolog(self):
        return mlflow.autolog()

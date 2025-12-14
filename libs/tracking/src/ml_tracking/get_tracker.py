from ml_tracking.ml_tracker import ML_Tracker


def get_tracker(
    tracker: str = "no_tracker",
) -> ML_Tracker:
    if tracker == "no_tracker":
        from ml_tracking.no_tracker import NoTracker

        return NoTracker("foo")

    elif tracker == "mlflow_tracker_azure":
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        from ml_tracking.mlflow_tracker import MlflowTracker

        ml_client = MLClient.from_config(
            credential=DefaultAzureCredential(),
        )
        workspace = ml_client.workspaces.get(ml_client.workspace_name)
        return MlflowTracker(workspace.mlflow_tracking_uri)
    else:
        raise NotImplementedError(f"Tracker {tracker} is not implemented.")

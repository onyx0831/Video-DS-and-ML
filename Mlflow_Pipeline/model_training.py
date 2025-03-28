import os
import set_dotenv  # NOQA
import tempfile
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from logging_config import setup_logger
logger = setup_logger(__name__)


def run_training(experiment_name: str) -> str:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.get_experiment_by_name(experiment_name)
    # 実験が存在しない場合は新規作成、あればその実験に紐づける
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name="model_training") as run:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'feature_engineering'",
            order_by=["start_time DESC"],
            max_results=1
        )
        feature_run_id = runs[0].info.run_id

        X_train_path = mlflow.artifacts.download_artifacts(run_id=feature_run_id, artifact_path="features/X_train.csv")
        y_train_path = mlflow.artifacts.download_artifacts(run_id=feature_run_id, artifact_path="features/y_train.csv")

        X = pd.read_csv(X_train_path)
        y = pd.read_csv(y_train_path)

        model = RandomForestClassifier()
        model.fit(X, y.values.ravel())

        # 一時ディレクトリにモデル保存してからログ
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model")
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="my_model")

        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("random_state", model.random_state)

        # 最新のバージョンを Production に昇格
        latest_versions = client.get_latest_versions(name="my_model", stages=["None"])
        if latest_versions:
            version = latest_versions[0].version
            client.transition_model_version_stage(
                name="my_model",
                version=version,
                stage="Production",
                archive_existing_versions=True  # 古いProductionはArchivedに
            )
            logger.info(f"Promoted model version {version} to Production.")

        return run.info.run_id
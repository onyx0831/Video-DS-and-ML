import os
import set_dotenv  # NOQA
import tempfile
import mlflow
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from logging_config import setup_logger
logger = setup_logger(__name__)


def run_feature_engineering(experiment_name: str) -> str:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.get_experiment_by_name(experiment_name)
    # 実験が存在しない場合は新規作成、あればその実験に紐づける
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name="feature_engineering") as run:
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 一時ディレクトリに保存
        with tempfile.TemporaryDirectory() as temp_dir:
            feature_dir = os.path.join(temp_dir, "features")
            os.makedirs(feature_dir, exist_ok=True)

            X_train.to_csv(os.path.join(feature_dir, "X_train.csv"), index=False)
            X_test.to_csv(os.path.join(feature_dir, "X_test.csv"), index=False)
            y_train.to_csv(os.path.join(feature_dir, "y_train.csv"), index=False)
            y_test.to_csv(os.path.join(feature_dir, "y_test.csv"), index=False)

            # アーティファクトとしてMLflowにアップロード
            mlflow.log_artifacts(feature_dir, artifact_path="features")
            logger.info(f"Uploaded feature engineering artifacts to MLflow.")

        return run.info.run_id

import os
import set_dotenv  # NOQA
import tempfile
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score
from logging_config import setup_logger
logger = setup_logger(__name__)


def run_predict(experiment_name: str):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.get_experiment_by_name(experiment_name)
    # 実験が存在しない場合は新規作成、あればその実験に紐づける
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name="predict") as run:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'feature_engineering'",
            order_by=["start_time DESC"],
            max_results=1
        )
        feature_run_id = runs[0].info.run_id

        X_test_path = mlflow.artifacts.download_artifacts(run_id=feature_run_id, artifact_path="features/X_test.csv")
        y_test_path = mlflow.artifacts.download_artifacts(run_id=feature_run_id, artifact_path="features/y_test.csv")

        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)

        model_uri = "models:/my_model/Production"
        model = mlflow.sklearn.load_model(model_uri)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("prediction_accuracy", acc)

        # 一時ディレクトリに予測結果を保存
        with tempfile.TemporaryDirectory() as temp_dir:
            preds_path = os.path.join(temp_dir, "predictions.csv")
            pd.DataFrame({"prediction": preds}).to_csv(preds_path, index=False)
            mlflow.log_artifact(preds_path)

        logger.info(f"Prediction accuracy: {acc:.4f}")
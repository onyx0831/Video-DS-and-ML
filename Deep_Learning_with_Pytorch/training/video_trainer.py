from typing import Callable, Dict
import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import EvalPrediction
# from sklearn.metrics import mean_squared_error


class SaveEvalPrediction:
    """
    Trainerクラスの初期化引数 compute_metrics に渡すためのクラス
    メトリクス計算 + pred-trueテーブルの保存 を行う
    """

    def __init__(
        self, compute_metrics: Callable[[EvalPrediction], Dict], output_dir: str
    ):
        """
        Args:
            output_dir: eval_prediction.csv の出力先
            compute_metrics: メトリクスを計算する関数。
                             単独でTrainerクラスの初期化引数 compute_metrics に渡せる形
        """
        self.n_eval = 1
        self.output_dir = output_dir
        self.output_fname = output_dir + "/eval_prediction.csv"
        self.compute_metrics = compute_metrics

    def __call__(self, ep: EvalPrediction) -> Dict:
        """
        1. validationのtrue-predテーブルを保存する
        2. メトリクスを計算する
        Args:
            ep: transformers.EvalPrediction（Trainerから自動的に渡される）
        Return:
            メトリクスの辞書
        """
        # DataFrameを作成
        # TODO: pred, trueがマルチな場合を考慮していない
        # df = pd.DataFrame({k: v.flatten() for k, v in ep._asdict().items()})
        dict_ep = {"predictions": ep.predictions, "label_ids": ep.label_ids}
        df = pd.DataFrame({k: v.flatten() for k, v in dict_ep.items()})
        df.insert(0, "n_eval", self.n_eval)
        # csv書き込み or 追記（初回以外）
        kwargs = {}
        if self.n_eval != 1:
            kwargs = dict(mode="a", header=False)
        df.to_csv(self.output_fname, **kwargs)
        self.n_eval += 1
        # メトリクス計算
        return self.compute_metrics(ep)

def custom_compute_metrics(res: EvalPrediction) -> Dict:
    # res.predictions, res.label_idsはnumpyのarray
    pred = res.predictions.argmax(axis=1)
    target = res.label_ids
    precision = precision_score(target, pred, average='macro')
    recall = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_training_history(trainer_state_path: str) -> pd.DataFrame:
    """
    transformers.Trainerクラスの.save_state()で保存されるtrainer_state.jsonを
    epochとstepを軸にcsvファイルの形式に変換する関数
    Args:
        trainer_state_path: trainer_state.jsonの相対パスを指定する
    Return:
        集計後のデータフレーム
    """
    with open(trainer_state_path, "r") as json_file:
        trainer_state = json.load(json_file)

    history_df = pd.DataFrame(trainer_state["log_history"])
    history_df = history_df.groupby(["epoch", "step"]).sum().reset_index()
    history_df = history_df[
        [
            "epoch",
            "step",
            "learning_rate",
            "loss",
            "eval_f1",
            "eval_loss",
            "eval_precision",
            "eval_recall",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
        ]
    ]
    return history_df
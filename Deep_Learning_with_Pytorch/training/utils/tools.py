import os
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curves(
    training_history: pd.DataFrame, save_dir: Optional[str] = None
) -> None:
    """
    lossとcorrelationの学習曲線をプロットする
    Args:
        training_history: カラムに'epoch', 'loss', 'eval_loss', 'eval_corr'を含むデータフレーム
        save_dir: 図を保存するディレクトリのパス。Noneなら保存しない。
    """
    x = training_history["epoch"].to_numpy()
    train_loss = training_history["loss"].to_numpy()
    eval_loss = training_history["eval_loss"].to_numpy()
    eval_corr = training_history["eval_corr"].to_numpy()

    # loss
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, train_loss, color="blue", linestyle="-", label="train loss")
    plt.plot(x, eval_loss, color="green", linestyle="--", label="val loss")
    plt.xlim(0, np.max(x))
    plt.legend(fontsize=12)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.title("Training and validation loss", fontsize=14)
    plt.grid()
    plt.tick_params(labelsize=12)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "loss.png"), bbox_inches="tight")
    # plt.show()
    plt.close()

    # corr
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, eval_corr, color="green", linestyle="--", label="val corr")
    plt.xlim(0, np.max(x))
    plt.legend(fontsize=12)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("correlation", fontsize=14)
    plt.title("Validation correlation coefficient", fontsize=14)
    plt.grid()
    plt.tick_params(labelsize=12)
    if save_dir:
        fig.patch.set_facecolor("white")
        plt.savefig(os.path.join(save_dir, "corr.png"), bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_scatter(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    save_dir: Optional[str] = None,
    score_name: str = "CTR",
    file_name: str = "scatter.png",
    corr: Optional[float] = None,
    rotation: int = 45,
):
    """
    スコア(CTR, Dscore)の予測と正解の散布図を作成する
    Args:
        y_pred: 予測値
        y_true: 正解値
        save_dir: 図を保存するディレクトリのパス。Noneなら保存しない。
        score_name: ラベルに表示するスコアの名前
        file_name: 図を保存する場合のファイル名
        corr: 図のタイトルに表示する相関係数の値。Noneなら表示しない。
        rotation: x軸の目盛りの数値が重ならないように回転する場合の角度。
    """
    min_v = min(np.min(y_pred), np.min(y_true))
    max_v = min(np.max(y_pred), np.max(y_true))
    fig = plt.figure(figsize=(5, 5))
    plt.xlim(min_v, max_v)
    plt.ylim(min_v, max_v)
    plt.plot([min_v, max_v], [min_v, max_v], linewidth=1, c="gray")
    plt.scatter(y_pred, y_true, color="b", s=5, linewidth=0, alpha=0.5)
    plt.xlabel(f"predicted {score_name}", fontsize=16)
    plt.ylabel(f"true {score_name}", fontsize=16)
    plt.xticks(fontsize=13, rotation=rotation)
    plt.yticks(fontsize=13)
    if corr is not None:
        plt.title("Correlation coefficient: {:.3f}".format(corr), fontsize=18)
    if save_dir:
        fig.patch.set_facecolor("white")
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    # plt.show()
    plt.close()
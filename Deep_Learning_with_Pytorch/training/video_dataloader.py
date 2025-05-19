from transformers.data import DefaultDataCollator


class VideoCollator(DefaultDataCollator):
    """
    videoデータのCollator
    （datasetから取り出した辞書型データのリストをbatch化するクラス）
    path -> Tensor への変換はDataset側でやっているので、デフォルトのバッチ結合だけを行う。
    以下のようなことをしたい場合は、このクラスを変更する。
        - 異なる画像サイズを組み合わせてpadding
        - RandomClip や Flipping などのランダム操作
    """

    pass
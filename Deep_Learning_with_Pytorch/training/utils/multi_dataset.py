from torch.utils.data import Dataset
from collections.abc import Sequence, Mapping


class MultiDataset(Dataset):
    """
    複数の Dataset をまとめてDataset にするモジュール
    Dataset のリスト、ディクショナリ もしくはそれらの再帰的な構造を受け付け、
    __getitem__ で、同じ構造として中身を取得できる

    例:
        d = MultiDataset({'image': Dataset(), 'meta': {'cmeta': Dataset(), 'qmeta': Dataset()}})
        d[0]
        -> {'image': Tensor(...), 'meta': {'cmeta': Tensor(...), 'qmeta': Tensor(...)}}

    extractor:
        __getitem__で取得する際に介する関数
        例: MultiDataset({text: {input_ids: DS()}, target: {target: DS()}},
                         extractor=lambda x: {**x['text'], **x['target']})
            -> {input_ids: Tensor(...), target: Tensor(...)}
    """

    def __init__(self, datasets, extractor=None):
        self.datasets = datasets
        self.__length = None
        self.__validate_recursively(self.datasets)
        self.extractor = extractor

    def __validate_recursively(self, datasets):
        """
        再帰的にデータセットを持つ構造かをチェックする
        同時に、データ長が同じかもチェックし、そのデータ長を self.__length に格納する

        正しい例:
            - dataset = Dataset()
            - dataset = [Dataset(), Dataset()]
            - dataset = {'image': Dataset(), 'meta': Dataset()}
            - dataset = {'image': Dataset(), 'meta': {'cmeta': Dataset(), 'qmeta': Dataset()}}
        ダメな例:
            - dataset = np.array, pd.DataFrame or torch.Tensor (or list of them)
        """
        if isinstance(datasets, Dataset):
            if self.__length is None:
                self.__length = len(datasets)
            else:
                if len(datasets) != self.__length:
                    raise ValueError("datasets must have same length.")
        elif isinstance(datasets, Sequence):
            for d in datasets:
                self.__validate_recursively(d)
        elif isinstance(datasets, Mapping):
            for d in datasets.values():
                self.__validate_recursively(d)
        else:
            raise ValueError("datasets must be recursive structure having Dataset.")

    def __len__(self):
        """__init__() -> __validate_recursively() で設定されたデータ長を返す"""
        assert self.__length is not None
        return self.__length

    def __getitem_recursively(self, datasets, index):
        """再帰的に中身を取り出す。"""
        if isinstance(datasets, Dataset):
            return datasets[index]
        elif isinstance(datasets, Sequence):
            T = type(datasets)
            return T([self.__getitem_recursively(d, index) for d in datasets])
        elif isinstance(datasets, Mapping):
            T = type(datasets)
            return T(
                {k: self.__getitem_recursively(v, index) for k, v in datasets.items()}
            )
        else:
            # __validate_recursively でチェックしているので、ここに来ることはないはず
            raise ValueError("datasets must be recursive structure having Dataset.")

    def __getitem__(self, index):
        """中身を取り出す。 実装本体は__getitem_recursively()."""
        if self.extractor is None:
            return self.__getitem_recursively(self.datasets, index)
        else:
            return self.extractor(self.__getitem_recursively(self.datasets, index))



################################################################################

if __name__ == "__main__":

    import pytest
    import numpy as np

    class ConstData(Dataset):
        def __init__(self, v, n=100):
            self.n, self.v = n, v

        def __len__(self):
            return self.n

        def __getitem__(self, _):
            return self.v

    d1 = ConstData(n=100, v=1)
    d2 = ConstData(n=100, v=2)
    d3 = ConstData(n=100, v=3)
    de = ConstData(n=101, v=1)

    # single dataset
    d = MultiDataset(d1)
    assert len(d) == 100
    assert d[0] == 1

    # list of dataset
    d = MultiDataset([d1, d2])
    assert len(d) == 100
    assert d[1] == [1, 2]

    # recursive dict of dataset
    d = MultiDataset({"1": d1, "23": (d2, d3)})
    assert len(d) == 100
    assert d[2] == {"1": 1, "23": (2, 3)}

    # another structure
    with pytest.raises(ValueError):
        d = MultiDataset(np.ones(shape=100))

    # different length
    with pytest.raises(ValueError):
        d = MultiDataset([d1, de])

    class ConstDictData(Dataset):
        def __init__(self, k, v, n=100):
            self.n, self.v = n, v
            self.k = k

        def __len__(self):
            return self.n

        def __getitem__(self, _):
            return {self.k: self.v}

    da = ConstDictData("a", 1, n=10)
    db = ConstDictData("b", 2, n=10)
    dm = MultiDataset([da, db], extractor=lambda x: {**x[0], **x[1]})
    assert dm[0]["a"] == 1
    assert dm[0]["b"] == 2

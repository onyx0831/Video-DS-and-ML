from torchvision import transforms
from torchvision.transforms import Normalize

from .imagegroup_processor import (
    GroupResize,
    GroupToTensor,
    Stack,
)


class VideoTransform:
    """
    動画を画像にした画像ファイルの前処理クラス
    動画を画像に分割しているため、分割された画像たちをまとめて前処理する点に注意
    """

    def __init__(self, resize, mean, std, fp16=False):
        self.data_transform = transforms.Compose(
            [
                GroupResize(int(resize)),  # 画像をまとめてリサイズ　resize*resizeになる
                GroupToTensor(fp16=fp16),  # データをPyTorchのテンソルに
                Stack(),  # 複数画像をframes次元で結合させる
                Normalize(mean, std),  # データを標準化
            ]
        )

    def __call__(self, img_group):

        return self.data_transform(img_group)

    def __str__(self):
        return self.__class__.__name__.lower()[:-9]
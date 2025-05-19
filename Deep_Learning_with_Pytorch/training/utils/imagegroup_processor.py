from PIL import Image
import numpy as np
import torch

from torchvision.transforms import Resize, CenterCrop


class GroupResize:
    """
    画像をまとめてリスケールするクラス。
    一辺が指定したサイズの正方形になる。
    """

    def __init__(self, resize, interpolation=Image.BILINEAR):
        """リスケールする処理"""
        self.rescaler = Resize(
            size=(resize, resize),
            interpolation=interpolation,
            antialias=True,
        )

    def _to_pil(self, img):
        """np.ndarray を PIL.Image に変換"""
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                return Image.fromarray(img, mode="L")
            elif img.shape[2] == 1:
                return Image.fromarray(img.squeeze(-1), mode="L")
            elif img.shape[2] == 3:
                return Image.fromarray(img, mode="RGB")
            elif img.shape[2] == 4:
                return Image.fromarray(img, mode="RGBA")
            else:
                raise ValueError(f"Unsupported ndarray shape: {img.shape}")
        elif isinstance(img, Image.Image):
            return img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    def __call__(self, img_group):
        """リスケールをimg_group(リスト)内の各imgに実施"""
        return [self.rescaler(self._to_pil(img)) for img in img_group]


class GroupCenterCrop:
    """
    画像をまとめてセンタークロップするクラス。
    （crop_size, crop_size）の画像を切り出す。
    """

    def __init__(self, crop_size):
        """センタークロップする処理"""
        self.ccrop = CenterCrop(crop_size)

    def __call__(self, img_group):
        """センタークロップをimg_group(リスト)内の各imgに実施"""
        return [self.ccrop(img) for img in img_group]


class _ToTensor:
    """
    ToTensorはオーバーヘッドが大きく処理が重いため、こちらで代用
    (PILToTensorでも一部改善されるが、それでも遅い)
    """

    def __init__(self, fp16=False):
        """
        Args:
            fp16: torch.float16 を返す
            * 内部的には、img->np.float16->(自動的に)torch.float16
        """
        self.np_dtype = np.float32 if not fp16 else np.float16

    def __call__(self, img) -> torch.Tensor:
        """
        ToTensorで行われている処理を、不要なオーバーヘッドなしで実施
        - PIL.Image または np.ndarray を受け付ける
        - [0, 1, ..., 255] -> 0~1 に変換
        - (H, W, C) → (C, H, W)
        """
        if isinstance(img, Image.Image):
            arr = np.array(img, dtype=self.np_dtype)
        elif isinstance(img, np.ndarray):
            if img.dtype != self.np_dtype:
                arr = img.astype(self.np_dtype)
            else:
                arr = img
        else:
            raise TypeError(
                f"Unsupported type: {type(img)}. Expected PIL.Image or np.ndarray."
            )

        # shapeチェックと変換
        if arr.ndim == 2:  # Grayscale: (H, W) → (1, H, W)
            arr = arr[:, :, None]
        elif arr.ndim != 3:
            raise ValueError(f"Invalid image shape: {arr.shape}. Expected 3D array.")

        return torch.as_tensor(arr.transpose(2, 0, 1) / 255.0)


class GroupToTensor:
    """画像をまとめてテンソル化するクラス。"""

    def __init__(self, fp16=False):
        """テンソル化する処理"""
        self.to_tensor = _ToTensor(fp16)

    def __call__(self, img_group):
        """
        テンソル化をimg_group(リスト)内の各imgに実施
        """

        return [self.to_tensor(img) for img in img_group]


class Stack:
    """
    画像を一つのテンソルにまとめるクラス。
    """

    def __call__(self, img_group):
        # img_groupはtorch.Size([3, 224, 224])を要素とするリスト
        return torch.stack(img_group, dim=0)  # frames次元(dim=0)を新たに作り結合

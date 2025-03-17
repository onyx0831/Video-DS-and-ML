import dataclasses
import torch
from .common import RiseSystemConfig

__all__ = [
    "MaskProcessorConfig",
    "MaskProcessor",
    "GrayMaskProcessorConfig",
    "GrayMaskProcessor",
    "MonotoneMaskProcessorConfig",
    "MonotoneMaskProcessor",
]


# --------------------------------------------------------------------------------
# base classes


@dataclasses.dataclass(frozen=True)
class MaskProcessorConfig:
    """
    マスク処理における共通かつstaticな設定

    Attributes:
        target_class: 対応するRiseModuleクラス名
                      (MaskProcessorのfrom_config(factory)で使用)
    """

    target_class: str = "MaskProcessor"  # forward参照, overridable


class MaskProcessor:
    """
    Mask処理の基底クラス
    Mask処理において共通して必要な変数・メソッドは、ここで管理・制約化する
    """

    def _process(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """マスク処理の具体的な実装はサブクラスで行う"""
        raise NotImplementedError

    def process(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        inputsにマスク処理をして返す
        実装本体はサブクラスの_process()
        Args:
            inputs: マスク処理対象の画像 (torch.Tensor, shape=(C, H, W))
            mask: マスク画像 (torch.Tensor, shape=(n, 1, H, W))
        Return:
            マスク処理された画像 (torch.Tensor, shape=(n, C, H, W))
        """
        masked_inputs = self._process(inputs, mask)
        assert masked_inputs.shape == (mask.shape[0], *inputs.shape)
        return masked_inputs

    @classmethod
    def from_config(
        cls,
        config: MaskProcessorConfig,
        system_config: RiseSystemConfig = RiseSystemConfig(),
    ):
        """
        configからMaskProcessorを作成するfactoryメソッド
        """
        target_class: type = globals()[config.target_class]
        return target_class(config, system_config)


# --------------------------------------------------------------------------------
# gray masking


@dataclasses.dataclass(frozen=True)
class GrayMaskProcessorConfig(MaskProcessorConfig):
    """
    Grayマスク処理においてstaticな設定
    共通の設定は、MaskProcessorConfigクラス参照
    """

    target_class: str = "GrayMaskProcessor"  # forward参照


class GrayMaskProcessor(MaskProcessor):
    """mask値に応じて、グレーで画像を隠す処理を行うクラス"""

    def __init__(
        self,
        config: GrayMaskProcessorConfig = GrayMaskProcessorConfig(),
        system_config: RiseSystemConfig = RiseSystemConfig(),
    ):
        self.config = config
        self.system_config = system_config

    def _process(self, inputs, mask):
        """
        グレーで画像を隠すようなマスク処理を行う
        (実際の処理としては、前処理済み画像にmaskを積算するだけ)
        """
        return inputs * mask


# --------------------------------------------------------------------------------
# monotonize


@dataclasses.dataclass(frozen=True)
class MonotoneMaskProcessorConfig(MaskProcessorConfig):
    """
    Monotoneマスク処理においてstaticな設定
    共通の設定は、MaskProcessorConfigクラス参照
    """

    target_class: str = "MonotoneMaskProcessor"  # forward参照


class MonotoneMaskProcessor(MaskProcessor):
    """
    mask値に応じて、Monotoneに置き換える処理を行うクラス
    (processorも柔軟に取り替えできる、というアピールの意味合いが強く、性能は未検証）
    """

    def __init__(
        self,
        config: GrayMaskProcessorConfig = GrayMaskProcessorConfig(),
        system_config: RiseSystemConfig = RiseSystemConfig(),
    ):
        self.config = config
        self.system_config = system_config

    def _process(self, inputs, mask):

        # RGBチャネルに対する重み（一般的に使用される値）
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=inputs.device).view(
            1, 3, 1, 1
        )

        # 各チャネルに重みを適用して和を取ることでグレースケール画像を取得
        grayscaled_inputs = (inputs * weights).sum(dim=1, keepdim=True)

        # マスクを適用
        masked_image = mask * inputs + (1 - mask) * grayscaled_inputs
        return masked_image

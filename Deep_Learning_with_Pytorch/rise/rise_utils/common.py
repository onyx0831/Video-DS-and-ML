import dataclasses
import torch

__all__ = [
    "RiseSystemConfig",
    "RiseModuleConfig",
    "RiseModule",
]


# --------------------------------------------------------------------------------
# common classes


@dataclasses.dataclass
class RiseSystemConfig:
    """
    RISE全体において横断的に関わる設定
    部分的な機能を提供する RiseModule(Config)とは別もの
    (役割も継承関係上も)
    入力に対して対応するため、これだけ frozen=Trueではない
    Attributes:
        device: 生成するテンソルのデバイス
        dtype: テンソルの型 (inputs.dtypeを想定)
    """

    dtype: torch.dtype = torch.float
    device: torch.device = torch.device("cpu")


# --------------------------------------------------------------------------------
# base abstract classes


@dataclasses.dataclass(frozen=True)
class RiseModuleConfig:
    """
    生成するRiseModuleの設定を保持するデータクラス、の抽象基底クラス

    Attributes:
        target_class: 対応するRiseModuleクラス名
                      (RiseModuleのfrom_config(factory)で使用)
    """

    target_class: str = "RiseModule"


class RiseModule:
    """
    Riseの部分的な機能を提供するクラス、の抽象基底クラス
    """

    def __init__(self, config: RiseModuleConfig, system_config: RiseSystemConfig):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: RiseModuleConfig, system_config: RiseSystemConfig):
        """
        configからAggregatorを作成するfactoryメソッド
        """
        target_class: type = globals()[config.target_class]
        return target_class(config, system_config)

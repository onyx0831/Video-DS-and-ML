import dataclasses
from typing import List, Union, Callable
from scipy.stats import t
import torch
from .common import RiseSystemConfig

__all__ = [
    "Aggregator",
    "AggregatorConfig",
    "ConditionalMeanAggregator",
    "ConditionalMeanAggregatorConfig",
    "LinearRegressionAggregator",
    "LinearRegressionAggregatorConfig",
]


# --------------------------------------------------------------------------------
# base classes


@dataclasses.dataclass(frozen=True)
class AggregatorConfig:
    """
    集約処理における共通かつstaticな設定
    現時点では特に設定項目はない

    Attributes:
        target_class: 対応するRiseModuleクラス名
                      (Aggregatorのfrom_config(factory)で使用)
    """

    target_class: str = "Aggregator"  # forward参照, overridable


class Aggregator:
    """
    集約処理の基底クラス (作りが中途半端だがとりあえず定義)
    inner_batchごとにupdateを呼び出し、
    最後にgetで集約された結果を返す、といった使い方を想定
    """

    def reset(self):
        raise NotImplementedError

    def update(self, mask: torch.Tensor, logits: torch.Tensor) -> None:
        raise NotImplementedError

    def get(self) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        config: AggregatorConfig,
        system_config: RiseSystemConfig = RiseSystemConfig(),
    ):
        """
        configからAggregatorを作成するfactoryメソッド
        """
        target_class: type = globals()[config.target_class]
        return target_class(config, system_config)


# --------------------------------------------------------------------------------
# conditional mean aggregator


@dataclasses.dataclass(frozen=True)
class ConditionalMeanAggregatorConfig(AggregatorConfig):
    """
    マスク生成においてstaticな設定
    現時点では特に設定項目はない
    共通の設定は、AggregatorConfigクラス参照
    """

    target_class: str = "ConditionalMeanAggregator"  # forward参照


class ConditionalMeanAggregator(Aggregator):
    """
    マスクされた領域の予測精度ダウン期待値を計算する
    """

    def __init__(
        self, config: ConditionalMeanAggregatorConfig, system_config: RiseSystemConfig
    ):
        self.config = config
        self.system_config = system_config
        self.reset()

    def reset(self):
        """
        初期化フラグを立てる
        (実際の初期化処理は、画像サイズ認識も兼ねて初回のupdate時に行われる)
        """
        self.initialize = True

    def __initialize(self, size, device):
        """
        実際に初期化処理を行う
        """
        self.fm_acc = torch.zeros(*size).to(device)  # (H, W)
        self.mask_acc = torch.zeros(*size).to(device)  # (H, W)
        self.list_logits_mean = []
        self.initialize = False

    def update(self, mask, logits):
        """
        batchごとに更新処理を行う
        Args:
            mask: マスク画像 (inner_batch_size, 1, H, W)
            logits: マスクをかけた画像に対する予測値 (inner_batch_size,)
        """
        if self.initialize:
            self.__initialize(mask.shape[-2:], mask.device)

        self.mask_acc += mask.mean(dim=0)[0]  # (H, W)
        # ポジネガスコアを計算
        # (f(I.m)-f(I)).(1-m)
        # paper5式の一部: f(I.m).m を改変
        # 可視化の大きさに大きな影響はないが、数字が直感的になる （マスクされた場合の予測精度ダウン期待値）
        fm = (1 - mask) * (logits - logits.mean()).reshape(-1, 1, 1, 1)
        self.fm_acc += fm.mean(dim=0)[0]  # (H, W)
        self.list_logits_mean.append(logits.mean())

    def get(self):
        """
        集約された結果を返す
        Return:
            S: マスクされた領域の予測精度ダウン期待値 (H, W)
        """
        if self.initialize:
            raise ValueError("This method must be called after update.")
        # 下がればpositiveなので、マイナスをつける
        return -self.fm_acc / self.mask_acc  # (H, W)

    def get_diff_logits(self, logits):
        """
        マスクした場合のlogits平均 - logits を返す
        Args:
            logits: マスクをかける前の入力画像に対する予測値
        Return:
            diff_logits: マスクした場合のlogits平均 - logits
        """
        if self.initialize:
            raise ValueError("This method must be called after update.")

        logits_mean = torch.Tensor(self.list_logits_mean).mean()
        return logits_mean - logits


# --------------------------------------------------------------------------------
# linear regression aggregator


@dataclasses.dataclass(frozen=True)
class LinearRegressionAggregatorConfig(AggregatorConfig):
    """
    マスク生成においてstaticな設定
    現時点では特に設定項目はない
    共通の設定は、AggregatorConfigクラス参照
    Args:
        get_value: getで返す値の種類
                   beta, t_value, p_value, signed_p_value の
                   いずれか、もしくはそのリスト
    """

    get_value: Union[str, List[str]] = "beta"
    target_class: str = "LinearRegressionAggregator"  # forward参照


class LinearRegressionAggregator(Aggregator):
    """
    マスクされた領域の予測精度ダウンの、マスク値に対する回帰係数を計算する
    """

    def __init__(
        self, config: LinearRegressionAggregatorConfig, system_config: RiseSystemConfig
    ):
        self.config = config
        self.system_config = system_config
        self.reset()

    def reset(self):
        """
        初期化フラグを立てる
        (実際の初期化処理は、画像サイズ認識も兼ねて初回のupdate時に行われる)
        """
        self.initialize = True

    def __initialize(self, size, device):
        """
        実際に初期化処理を行う
        """
        # 初期化
        self.n_samples = 0
        self.mean_x = torch.zeros(size, device=device)
        self.mean_y = torch.tensor(0.0, device=device)
        self.M2_x = torch.zeros(size, device=device)
        self.M2_xy = torch.zeros(size, device=device)
        self.rss = torch.zeros(size, device=device)
        self.initialize = False

    def update(self, mask, logits):
        """
        データストリームからバッチを順次読み込み、統計量を更新する。

        Args:
            mask (torch.Tensor): x_data に相当する (N, H, W) のテンソル。
            logits (torch.Tensor): y_data に相当する (N,) のテンソル。
        """
        if self.initialize:
            self.__initialize(mask.shape[-2:], mask.device)

        B = logits.shape[0]  # バッチサイズ
        mask = mask.squeeze(1)  # (N, H, W)

        # バッチの平均と二乗偏差和を計算
        mean_x_batch = torch.mean(mask, dim=0)
        mean_y_batch = torch.mean(logits)

        M2_x_batch = torch.sum((mask - mean_x_batch) ** 2, dim=0)
        M2_xy_batch = torch.sum(
            (mask - mean_x_batch) * (logits.view(-1, 1, 1) - mean_y_batch), dim=0
        )

        # 全体の平均と二乗偏差和を更新
        delta_x = mean_x_batch - self.mean_x
        delta_y = mean_y_batch - self.mean_y

        new_mean_x = self.mean_x + delta_x * B / (self.n_samples + B)
        new_mean_y = self.mean_y + delta_y * B / (self.n_samples + B)

        self.M2_x = (
            self.M2_x
            + M2_x_batch
            + delta_x**2 * self.n_samples * B / (self.n_samples + B)
        )
        self.M2_xy = (
            self.M2_xy
            + M2_xy_batch
            + delta_x * delta_y * self.n_samples * B / (self.n_samples + B)
        )

        # 平均を更新
        self.mean_x = new_mean_x
        self.mean_y = new_mean_y

        # 回帰係数の計算
        beta = self.M2_xy / self.M2_x

        # 残差平方和（RSS）の更新
        y_pred = self.mean_y + beta * (mask - self.mean_x)
        residual = logits.view(-1, 1, 1) - y_pred
        self.rss += torch.sum(residual**2, dim=0)

        # サンプル数を更新
        self.n_samples += B

    def get(self):
        """
        統計量を計算し、回帰係数、t値、p値, 符号付きの規格化したp値, を返す。

        Args:
            get_value (Union[str, List[str]]): 取得する値の種類
                "beta": 回帰係数
                "t_value": t値
                "p_value": p値
                "signed_p_value": 規格化したp値
                                  sign(beta) * (1 - p_value)
        Returns:
            return_values (List[torch.Tensor]): 取得した値
        """
        if self.initialize:
            raise ValueError("This method must be called after update.")

        # 分散の推定
        variance_x = self.M2_x / (self.n_samples - 1)
        sigma_squared = self.rss / (self.n_samples - 2)  # 自由度 n_samples - 2
        # 標準誤差の計算
        se_beta = torch.sqrt(sigma_squared / (self.n_samples * variance_x))
        # 回帰係数の計算
        beta = self.M2_xy / self.M2_x
        # t値の計算
        t_stat_beta = beta / se_beta
        # p値の計算
        p_value_beta = 2 * (
            1 - t.cdf(abs(t_stat_beta.cpu().numpy()), df=self.n_samples - 2)
        )
        p_value_beta = torch.Tensor(p_value_beta).to(t_stat_beta.device)
        signed_p_value_beta = torch.sign(beta) * (1 - p_value_beta)

        # 返り値の選択
        return_value_dict = dict(
            beta=beta,
            t_value=t_stat_beta,
            p_value=p_value_beta,
            signed_p_value=signed_p_value_beta,
        )
        if isinstance(self.config.get_value, str):
            return return_value_dict[self.config.get_value]
        else:
            return_values = []
            for value in self.config.get_value:
                return_values.append(return_value_dict[value])
            return return_values

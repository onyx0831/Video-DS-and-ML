import dataclasses
import warnings
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# private
from .common import RiseSystemConfig
from .sub_models import load_segment_model, load_depth_model
from ...functional.deprecated import deprecated

__all__ = [
    "MaskGeneratorConfig",
    "MaskGenerator",
    "UpsampledGridMaskGeneratorConfig",
    "UpsampledGridMaskGenerator",
    "GPMaskGeneratorConfig",
    "GPMaskGenerator",
]

# deprecated soon
__all__ += ["_generate_mask"]


# --------------------------------------------------------------------------------
# base classes


@dataclasses.dataclass(frozen=True)
class MaskGeneratorConfig:
    """
    マスク生成における共通かつstaticな設定

    Attributes:
        target_class: 対応するRiseModuleクラス名
                      (MaskGeneratorのfrom_config(factory)で使用)
        size: 画像サイズ(=maskサイズ)
              例: (224, 224)
    """

    target_class: str = "MaskGenerator"  # forward参照, overridable
    size: tuple = (224, 224)


class MaskGenerator:
    """
    Mask生成の基底クラス
    Mask生成において共通して必要な変数・メソッドは、ここで管理・制約化する
    """

    def _generate(self, n) -> torch.Tensor:
        """マスク生成の具体的な実装はサブクラスで行う"""
        raise NotImplementedError

    def generate(self, n) -> torch.Tensor:
        """
        マスクを生成して返す
        実装本体はサブクラスの_generate()
        Return:
            マスク（torch.Tensor, shape=(n, 1, size[0], size[1])）
        """
        mask = self._generate(n)
        assert mask.shape == (n, 1, *self.config.size)
        return mask.to(self.system_config.dtype)

    def set_image(self, image: Image.Image):
        """
        一部のGeneratorは、画像情報を使用するため、共通のメソッドを用意
        デフォルトでは何もしない
        """
        pass

    @classmethod
    def from_config(
        cls,
        config: MaskGeneratorConfig,
        system_config: RiseSystemConfig = RiseSystemConfig(),
    ):
        """
        configからMaskGeneratorを作成するfactoryメソッド
        """
        target_class: type = globals()[config.target_class]
        return target_class(config, system_config)


# --------------------------------------------------------------------------------
# upsampled grid mask


@dataclasses.dataclass(frozen=True)
class UpsampledGridMaskGeneratorConfig(MaskGeneratorConfig):
    """
    マスク生成においてstaticな設定
    共通の設定は、MaskGeneratorConfigクラス参照

    Attributes:
        mask_grid: maskの粒度。デフォルトは論文値
    """

    target_class: str = "UpsampledGridMaskGenerator"  # forward参照
    mask_grid: tuple = (7, 7)


class UpsampledGridMaskGenerator(MaskGenerator):
    """7x7のバイナリマスクを生成し、それを224x224にupsampleして生成するクラス"""

    def __init__(
        self,
        config: UpsampledGridMaskGeneratorConfig = UpsampledGridMaskGeneratorConfig(),
        system_config: RiseSystemConfig = RiseSystemConfig(),
    ):
        self.config = config
        self.system_config = system_config

    def _generate(self, n):
        """
        マスクを生成する
          - 7x7のバイナリマスクを生成し、それを224x224にupsampleする
          - ref: https://arxiv.org/pdf/1806.07421.pdf (5~6p)
        Args:
            n: マスクの枚数
        Return:
            マスク（torch.Tensor, shape=(n, 1, size[0], size[1])）
        """
        device, dtype = self.system_config.device, self.system_config.dtype
        config = self.config
        H, W = config.size
        h, w = config.mask_grid
        Ch, Cw = int(np.floor(H / h)), int(np.floor(W / w))

        # 1. (h, w)サイズの mask原型を作成  .float() は後のupsamplingのため
        M1 = torch.randint(2, size=(n, 1, h, w), device=device).float()
        # 2. (H, W） よりやや大きいサイズに upsample (bilinear補完)。 Upsamplingはbf16不可なのでfloatのみ
        M2 = F.interpolate(
            M1, size=((h + 1) * Ch, (w + 1) * Cw), mode="bilinear", align_corners=True
        )
        # 3. (H, W)サイズにカット
        crop = transforms.RandomCrop((H, W))
        M3 = torch.cat([crop(m).unsqueeze(0) for m in M2], dim=0).to(dtype)
        return M3


@deprecated(instead="UpsampledGridMask")
def _generate_mask(n, device, dtype=torch.float, size=(224, 224), mask_grid=(7, 7)):
    # 昔のやつ
    config = UpsampledGridMaskGeneratorConfig(size=size, mask_grid=mask_grid)
    system_config = RiseSystemConfig(device=device, dtype=dtype)
    mask = UpsampledGridMaskGenerator(config, system_config)
    return mask.generate(n)


################################################################################
# GP mask covariance functions


class CovarianceMixin:
    """
    共分散行列を算出するためのメソッドを提供するMixinクラス
    (関数で作ると色々とっ散らかるので、Mixinとしてまとめた)

    以下の変数を使用する
      - self.config: GPMaskGeneratorConfig
      - self.system_config: RiseSystemConfig

    継承先で以下のメソッドが使用可能になる
     - covariance(size, image)
    """

    def __calc_planar_distance2(self, size, image=None):
        """
        平面距離マトリクスを算出する
        Args:
            config: GPMaskGeneratorConfig
            size: 生成するマスクのサイズ
            image: np.ndarray, shape=(H, W, C)
        Return:
            平面距離2乗マトリクス: (size[0]*size[1], size[0]*size[1])
        """
        # config = self.config
        system_config = self.system_config

        # 計算
        xs = torch.linspace(0, size[0] / max(size), size[0]).to(system_config.device)
        ys = torch.linspace(0, size[1] / max(size), size[1]).to(system_config.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid_x_flat = grid_x.flatten()
        grid_y_flat = grid_y.flatten()

        # 距離計算のために、grid_x_flatとgrid_y_flatの各組み合わせの差を計算
        sq_dist_x = grid_x_flat[:, None] - grid_x_flat[None, :]
        sq_dist_y = grid_y_flat[:, None] - grid_y_flat[None, :]

        # 各点間のユークリッド距離の二乗
        sq_dist = sq_dist_x**2 + sq_dist_y**2
        return sq_dist

    def __calc_depth_distance2(self, size, image=None):
        """
        深度方向の距離2乗マトリクスを算出する
        Args:
            config: GPMaskGeneratorConfig
            size: 生成するマスクのサイズ
            image: np.ndarray, shape=(H, W, C)
        Return:
            深度方向の距離2乗マトリクス: (size[0]*size[1], size[0]*size[1])
        """
        config = self.config
        system_config = self.system_config
        s2 = size[0] * size[1]

        if config.depth_weight == 0:
            return torch.zeros(s2, s2, device=system_config.device)
        elif image is None:
            raise ValueError("image is required when depth_weight > 0.")

        # else ...
        # depth model で深度を推定
        depth_model = load_depth_model(system_config.device)
        # あまり良くないが、pipelineの仕様上ここだけPIL.Imageに変換して推論
        image = Image.fromarray(image)
        # depth = depth_model.infer_image(image)  # (H, W)
        depth = depth_model(image)["predicted_depth"]  # Tensor(1, H', W', cpu)

        # 次元を調整しつつ、リサイズ
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=size,
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )  # (1, 1, H, W)
        depth = depth[0, 0]  # (H, W)

        # 正規化とデバイス移行
        depth = (depth - depth.min()) / (depth.max() - depth.min())  # 正規化
        depth = depth.to(system_config.device)

        # 距離マトリクス化
        depth_flat = depth.flatten()
        row_matrix = torch.tile(depth_flat, (len(depth_flat), 1))
        col_matrix = row_matrix.T
        depth_distance2_matrix = (row_matrix - col_matrix) ** 2
        return depth_distance2_matrix

    def __covariance_by_distance(self, size, image=None):
        """
        距離に関する共分散マトリクスを算出する
        Args:
            size: 生成するマスクのサイズ
            image: np.ndarray, shape=(H, W, C)
        Return:
            距離に関する共分散マトリクス: (size[0]*size[1], size[0]*size[1])
        """
        config = self.config
        # system_config = self.system_config

        d2_planar = self.__calc_planar_distance2(size, image)
        d2_depth = self.__calc_depth_distance2(size, image)
        d2 = d2_planar + (config.depth_weight**2) * d2_depth
        # RBFカーネル関数を適用
        s2 = size[0] * size[1]
        return torch.exp(-0.5 * d2 / config.length_scale**2).reshape(s2, s2)

    def __covariance_by_segment(self, size, image=None):
        """
        セグメントに関する共分散マトリクスを算出する
        Args:
            size: 生成するマスクのサイズ
            image: np.ndarray, shape=(H, W, C)
        Return:
            セグメントに関する共分散マトリクス: (size[0]*size[1], size[0]*size[1])
        """
        config = self.config
        system_config = self.system_config
        s2 = size[0] * size[1]

        if config.segment_weight == 0:
            return torch.zeros(s2, s2, device=system_config.device)
        elif image is None:
            raise ValueError("image is required when segment_weight > 0.")

        seg_model = load_segment_model(system_config.device)

        segs = seg_model.generate(image)  # ~2.2[sec] (cached)
        # 1つもセグメントがない場合、共分散行列はゼロとして返す
        if len(segs) == 0:
            warnings.warn("No segment found. Return zero matrix.")
            return torch.zeros(s2, s2, device=system_config.device)

        segs = np.stack([seg["segmentation"].astype(np.float32) for seg in segs])

        # seg vectors を作成
        step_h = np.linspace(0, image.shape[0] - 1, size[0]).astype(int)
        step_w = np.linspace(0, image.shape[1] - 1, size[1]).astype(int)
        vs = (
            np.stack(segs, axis=-1)[step_h][:, step_w]
            .reshape(-1, len(segs))
            .astype(float)
        )

        # seg similarity を作成
        sim_mat = vs.dot(vs.T)
        return torch.Tensor(sim_mat).to(system_config.device)

    def covariance(self, size, image=None):
        """
        共分散マトリクスを算出する
        Args:
            size: 生成するマスクのサイズ
            image: np.ndarray, shape=(H, W, C)
        Return:
            共分散マトリクス: (size[0]*size[1], size[0]*size[1])
        """
        config = self.config
        system_config = self.system_config

        # 距離に基づく共分散行列
        cov_dist = self.__covariance_by_distance(size, image)
        # セグメントに基づく共分散行列
        cov_seg = self.__covariance_by_segment(size, image)

        # 重み付きの和
        cov = (
            1 - config.segment_weight
        ) * cov_dist + config.segment_weight * cov_seg
        # 正定値性を保つための微小値を加算
        cov += config.eps * torch.eye(cov.shape[0], device=system_config.device)
        return cov


################################################################################
# GP mask


@dataclasses.dataclass(frozen=True)
class GPMaskGeneratorConfig(MaskGeneratorConfig):
    """
    マスク生成においてstaticな設定
    共通の設定は、MaskGeneratorConfigクラス参照

    Additonal Args:
        downscale_factor: 内部的な画像サイズの縮小率（縦横共通）
                          制約: (H % factor) == (W % factor) == 0
        length_scale: マスクの距離スケール。（距離はsizeの長い方を1として計算）
        segment_weight: Segmentによる共分散行列を考慮する重み
                        制約: 0 ~ 1
        depth_weight: 深度方向の距離の重み
        sigma_scale: 0-1のマスク値に変換するときのスケール
        sigma_bias: 0-1のマスク値に変換するときのバイアス
        eps: 正定値性を保つための微小値
    """

    target_class: str = "GPMaskGenerator"  # forward参照
    downscale_factor: int = 4
    length_scale: float = 0.1
    segment_weight: float = 0.0
    depth_weight: float = 0.0
    sigma_scale: int = 2
    sigma_bias: int = 0
    eps: float = 1e-4


class GPMaskGenerator(MaskGenerator, CovarianceMixin):
    """共分散付きGaussによるマスクを生成するクラス"""
    def __init__(
        self,
        config: GPMaskGeneratorConfig = GPMaskGeneratorConfig(),
        system_config: RiseSystemConfig = RiseSystemConfig(),
    ):
        self.config = config
        self.system_config = system_config

        d = config.downscale_factor
        # 計算上の画像サイズ
        self.size_small = tuple(s // d for s in config.size)
        # 共分散行列のcholesky分解、Noneかどうかで計算の有無を判断
        self.cov_L = None
        # downscale_factorの制約条件
        assert config.size[0] % d == config.size[1] % d == 0

    def set_image(self, image: Image.Image):
        """
        画像の情報を設定する。
        image = None の場合も一括して対応

        具体的には、画像に対応する共分散行列と
        そのcholesky分解を計算する (O(W^3H^3) ~0.005[sec])
        """
        if image is not None:
            image = np.array(image.convert("RGB"))
        cov = self.covariance(self.size_small, image)
        cov_L = torch.linalg.cholesky(cov)
        self.cov_L = cov_L

    def _generate(self, n):
        """
        マスクを生成する
          - ref: https://www.notion.so/GP-RISE-3ba7f83de639490b98c7566a26a17e58
        """
        # パラメータ取得
        config = self.config
        device = self.system_config.device
        size = config.size
        sigma_scale = config.sigma_scale
        sigma_bias = config.sigma_bias
        size_small = self.size_small
        size_prod = size_small[0] * size_small[1]

        # 共分散行列が未計算の場合、計算
        if self.cov_L is None:
            image = None  # あえてね
            cov = self.covariance(self.size_small, image)
            cov_L = torch.linalg.cholesky(cov)
            self.cov_L = cov_L

        # 共分散に基づきサンプリング (O(NW^2H^2))
        samples = self.cov_L @ torch.randn(size_prod, n).to(device)

        # 低解像度サンプルを2Dに再形成
        samples_2d = samples.view(size_small[0], size_small[1], n, 1)
        samples_2d = samples_2d.permute(
            2, 3, 0, 1
        )  # (n, 1, size_small[0], size_small[1])

        # upsample
        samples_high_2d = F.interpolate(
            samples_2d, size=size, mode="bicubic", align_corners=True
        )  # (n, 1, size, size)

        # scale, biasで線形変換したあと、sigmoidで0-1に変換
        masks = 1 / (1 + torch.exp(-(sigma_scale * samples_high_2d + sigma_bias)))
        return masks

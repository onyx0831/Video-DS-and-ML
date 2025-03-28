from tqdm import tqdm
from PIL import Image
from typing import Callable, Union, Optional, Sequence
import torch
from torch import nn

from .rise_utils.mask_generator import (
    MaskGenerator,
    MaskGeneratorConfig,
    UpsampledGridMaskGeneratorConfig,
)
from .rise_utils.mask_processor import (
    MaskProcessor,
    MaskProcessorConfig,
    GrayMaskProcessorConfig,
)
from .rise_utils.aggregator import (
    Aggregator,
    AggregatorConfig,
    ConditionalMeanAggregatorConfig,
)
from .rise_utils.common import RiseSystemConfig
from .utils import recursive_lambda

__all__ = ["RISE"]


################################################################################
# RISE


class RISE:
    """
    RISEによる画像のポジネガ可視化を行うクラス
    詳細: https://arxiv.org/abs/1806.07421
    """

    def __init__(
        self,
        forward_func: Callable,
        process_after_masking: Callable = lambda x: x,
        mask_generator_config: MaskGeneratorConfig = UpsampledGridMaskGeneratorConfig(),
        mask_processor_config: MaskProcessorConfig = GrayMaskProcessorConfig(),
        aggregator_config: AggregatorConfig = ConditionalMeanAggregatorConfig(),
    ) -> None:
        """
        Args:
            forward_func: モデルのforward関数、もしくはそれに対応する関数
                          下記のようなInput, Outputであること
                Input: 前処理済み画像テンソルのバッチ
                       torch.Tensor, shape=(batch_size, n_channel, height, width)
                Output: 予測値 (logits)
                        torch.Tensor, shape=(batch_size,) or (batch_size, num_targets)
            process_after_masking: マスクをつけた後の処理関数
                                   この関数を通した後、forward_funcに渡される。
                Input: マスクが付与された画像データ
                       torch.Tensor, shape=(batch_size, n_channel, height, width)
            mask_generator_config: マスク生成器の設定
                                   default: UpsampledGridMaskGeneratorConfig()
                                   (RISEに準拠, グリッドマスクをアップサンプリング)
            mask_generator_config: マスク生成器の設定
                                   default: GrayMaskProcessorConfig()
                                   (RISEに準拠, マスク領域をグレーに塗る)
            aggregator_config: 集約処理器の設定
                               defaultでは ConditionalMeanAggregatorConfig()
                               (RISEにほぼ準拠, pixelごとにマスクされた場合のlogits平均を取る)
        """
        self.forward_func = forward_func
        self.process_after_masking = process_after_masking
        # create modules
        self.system_config = RiseSystemConfig()  # 各モジュールで共有・参照される
        self.mask_generator = MaskGenerator.from_config(
            mask_generator_config, system_config=self.system_config
        )
        self.mask_processor = MaskProcessor.from_config(
            mask_processor_config, system_config=self.system_config
        )
        self.aggregator = Aggregator.from_config(
            aggregator_config, system_config=self.system_config
        )

    def attribute(
        self,
        inputs: torch.Tensor,
        input_images: Optional[Sequence[Image.Image]] = None,
        additional_forward_args: Union[dict, list, tuple] = None,
        return_diff_logits: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        要因可視化を実行し、その結果を返す
        バッチデータに対して実行するが、各レコードの処理は __attribute_single で行う。

        Args:
            inputs: forward関数への入力データのバッチ
            input_images: 入力画像のリスト（バッチ）
                          一部のマスク生成器において必要
            additional_forward_args: (dict, list, tuple) マスクに応じて複数辞書objコピーを作りforward_funcに渡す
                    dictなら**、listやtupleなら*で渡す
        その他は、__attribute_single を参照
        """
        assert inputs.ndim == 4
        assert input_images is None or len(inputs) == len(input_images)

        # 各データに対してattribute_singleを実行
        outputs_attribute = []
        # 入力共通化のため、input_imagesがNoneの場合はNoneのlistを入れる
        if input_images is None:
            input_images = [None] * len(inputs)
        for i, (x, image) in enumerate(zip(inputs, input_images)):
            additional_forward_args_i = recursive_lambda(
                additional_forward_args, lambda t: t[i], ignore_rest=True
            )
            output = self.__attribute_single(
                x,
                image,
                additional_forward_args_i,
                return_diff_logits=return_diff_logits,
                *args,
                **kwargs,
            )
            outputs_attribute.append(output)

        if isinstance(outputs_attribute[0], (tuple, list)):
            return tuple(
                torch.stack([x[i] for x in outputs_attribute])
                for i in range(len(outputs_attribute[0]))
            )
        else:
            return torch.stack(outputs_attribute)

    def __attribute_single(
        self,
        inputs: torch.Tensor,
        input_image: Optional[Image.Image] = None,
        additional_forward_args: Union[dict, list, tuple] = None,
        target: int = None,
        n_samples: int = 2048,
        batch_size: int = 16,
        return_diff_logits: bool = False,
        show_progress: bool = False,
        leave_progressbar: bool = True,
    ) -> torch.Tensor:
        """
        要因可視化を実行し、その結果を返す

        Args:
            inputs: forward関数への入力データ
                    ここで与えるものが可視化対象になる。
                    torch.Tensor, shape=forward_funcの第一引数に準拠
                    batch_size の次元は含まない (attributeでsliceされている)
            input_image: 入力画像
                         一部のマスク生成器において必要
            additional_forward_args: 可視化対象以外の入力
                                     辞書、リスト、タプルはそれぞれ1回展開される
                                     batch_size の次元は含まない
            target: targetのid番号。
                    モデルoutputの列番号に対応
                    指定: 多クラス分類（マルチアウトプット）、
                    None: シングルアウトプットなモデル、を想定
            n_samples: モデルを回す試行回数
            batch_size: モデルを回す際の内部バッチサイズ
                        TODO: n_samples % batch_size != 0 のケースは未実装
            return_diff_logits: Trueの場合、"マスクした場合のlogits平均 - logits" を同時に出力
            show_progress: tqdmを表示する
            leave_progressbar:(bool) Trueならプログレスバーが終わっても残る。Falseなら消える。
                        注:書き出すログに影響はない。jupyterlabでは消えても空白な行が残る。
        Return:
            return_diff_logits=False の場合、A を、
            return_diff_logits=True の場合、tuple(A, B) を返す

            A. ポジティブ-ネガティブの要因可視化結果
                "+" の値の場合は、ポジティブ（該当領域をマスクすることで予測値が下がる）で、
                "-" の値の場合は逆。
                0がニュートラルである以外、特に正規化はしていない。
                torch.Tensor, shape=(height, width)

            B. マスクした場合のlogits平均 - マスクをかけない場合のlogits
                torch.Tensor (scalar)

        TODO: optionalで乱数固定
        """
        # 以下、未実装
        if n_samples % batch_size != 0:
            raise NotImplementedError

        assert inputs.ndim == 3

        # 入力データのデバイス、dtypeをsystem_configに設定
        # system_configは各モジュールにて参照される
        self.system_config.device = inputs.device
        self.system_config.dtype = inputs.dtype

        # マスクされたデータに対して予測を実行
        n_loop = n_samples // batch_size

        def _forward(x, additional_forward_args):
            """
            additional_forward_argsをbatch化し、追加引数の型に応じてforwardを適切に実行する
            外部変数: self.forward_func
            """
            batch_size = x.shape[0]
            args = recursive_lambda(
                additional_forward_args,
                lambda x: torch.stack([x] * batch_size),
                ignore_rest=True,
            )  # batch化
            with torch.no_grad():
                if isinstance(args, (tuple, list)):
                    logits = self.forward_func(x, *args)
                elif isinstance(args, (dict,)):
                    logits = self.forward_func(x, **args)
                elif args is None:
                    logits = self.forward_func(x)
                else:
                    logits = self.forward_func(x, args)
            return logits.detach()  # (inner_batch_size,)

        # マスク生成器の設定
        self.mask_generator.set_image(input_image)
        # 集約器の初期化
        self.aggregator.reset()

        # マスクをかけたデータをモデルに通す.. を繰り返す
        loop = range(n_loop)
        if show_progress:
            loop = tqdm(
                loop, total=n_loop, desc="[rise.single]", leave=leave_progressbar
            )
        for _ in loop:
            # バッチごとにマスク処理
            mask = self.mask_generator.generate(n=batch_size)
            x = self.mask_processor.process(inputs, mask)  # (inner_batch_size, C, H, W)
            x = self.process_after_masking(x)
            # マスクのかかったデータをモデルに通す
            logits = _forward(x, additional_forward_args)  # (inner_batch_size,)
            # 多クラス分類の場合、target番目の列を取り出す
            if target is not None:
                logits = logits[:, target]
            self.aggregator.update(mask=mask, logits=logits)

        S = self.aggregator.get()
        if not return_diff_logits:
            return S
        else:
            logits_true = _forward(inputs.unsqueeze(0), additional_forward_args)[0]
            return S, self.aggregator.get_diff_logits(logits=logits_true)

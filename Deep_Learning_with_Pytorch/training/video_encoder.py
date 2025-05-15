import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Callable, List

from models.video_mae.model import VideoMAE


class PredictionHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: List[int] = [], output_dim: int = 1
    ):
        """
        Args:
            input_dim (int): Input dimension
            hidden_dims (List[int], optional): MLP hidden unit dimension.
                Defaults to [], the module becomes a linear model.
            output_dim (int, optional): Output dimension. Defaults to 1.
        """
        super().__init__()
        layers = []
        n_in = input_dim
        for u in hidden_dims:
            n_out = u
            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.ReLU())
            n_in = n_out
        layers.append(nn.Linear(n_in, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.mlp(input_values)
    

class Encoder(nn.Module):
    """
    動画フレームを入力としたエンコーダー
    """
    def __init__(
        self,
        model_name: str,
        model_kwargs: dict = {},
        pretrained: bool = True,
        output_dim: int = 1,
        head_hidden_dims: List[int] = [],
        loss_fct: Optional[Callable] = None,
        last_activation: Optional[Callable] = nn.Sigmoid(),
        post_loss_process: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.model = encoder_selector(model_name, pretrained, model_kwargs)
        self.encoder_dim = self.model.encoder_dim
        self.head = PredictionHead(
            input_dim=self.encoder_dim,
            hidden_dims=head_hidden_dims,
            output_dim=output_dim,
        )

        self.loss_fct = loss_fct
        if last_activation is None:
            self.last_activation = nn.Identity()
        else:
            self.last_activation = last_activation

        if post_loss_process is None:
            self.post_loss_process = nn.Identity()
        else:
            self.post_loss_process = post_loss_process

    def forward(
        self,
        input_values: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Args:
            input_values (torch.Tensor): 動画フレームの特徴量 shape=(batch_size, frame, 3, H, W)
            frame_mask (Optional[torch.BoolTensor], optional): フレームのマスク shape=(batch_size, frame)
                                                           ViTベースのモデルでのみ有効。1ならマスクする。0ならマスクしない。
            targets (Optional[torch.Tensor], optional): ターゲット shape=(batch_size,)
        Returns:
            ModelOutput。以下のメンバ変数を持つ。
                - encoded_state: エンコーダを代表するベクトル表現
                                 torch.Tensor, shape=(batch_size, encoder_dim)
                - logits: 予測値
                          torch.Tensor, shape=(batch_size, output_dim)
                - loss: バッチに対するlossの平均値
                        torch.Tensor, shape=(1,)
                - last_hidden_state: 最終層の全シーケンス表現。
                                     torch.Tensor, shape=(batch_size, seq_length, encoder_dim)
                - atentions: 全層のアテンション(モデルによってはNoneが返される。tesorの場合でもモデルによってshapeが異なる)
        """
        output = self.model(input_values, frame_mask)

        encoded_state = output.encoded_state
        logits = self.last_activation(self.head(encoded_state))
        loss = None
        if targets is not None and self.loss_fct is not None:
            loss = self.loss_fct(logits, targets.squeeze_())
        logits = self.post_loss_process(logits)

        return ModelOutput(
            encoded_state=encoded_state,
            logits=logits,
            loss=loss,
            last_hidden_state=output.last_hidden_state,
            attentions=output.attentions,
        )


def encoder_selector(model_name: str, pretrained: bool = True, model_kwargs: dict = {}):
    """
    model_nameに応じたエンコーダーを返す
    サポートされているmodel_name:
        - VideoMAE: VideoMAEモデル https://huggingface.co/MCG-NJU/videomae-base
    """
    if model_name == "VideoMAE":

        model = VideoMAE(pretrained=pretrained, **model_kwargs)
    else:
        raise ValueError(f"model_name: {model_name} is not supported.")
    return model
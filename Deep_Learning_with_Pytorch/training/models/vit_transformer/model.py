import torch
import torch.nn as nn
from torchvision import models
from einops import rearrange
from typing import Callable, Optional
from transformers.modeling_outputs import ModelOutput


class Attention(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.linear_att = nn.Linear(self.input_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (b, 15, input_dim)
        att = self.linear_att(x).squeeze(-1)
        alpha = self.softmax(att)  # (b, 15)
        attention_weighted_encoding = torch.sum(
            alpha.unsqueeze(-1) * x, dim=1
        )  # (b, input_dim)
        return attention_weighted_encoding, alpha


class UniFrameNet(nn.Module):
    def __init__(
        self,
        premodel_name: str = "resnet50",
        output_dim: int = 256,
        isfreeze: bool = False,
    ) -> None:  # resnet or vit
        super().__init__()

        if premodel_name == "resnet50":
            self.premodel = models.resnet50(pretrained=True)

        elif premodel_name == "resnet101":
            self.premodel = models.resnet101(pretrained=True)

        elif premodel_name == "resnet152":
            self.premodel = models.resnet152(pretrained=True)

        else:
            # todo:vitモデル対応、暫定的にresnet50にしてる
            # self.premodel = ViTModel.from_pretrained(
            # "google/vit-base-patch16-224-in21k"
            # )
            self.premodel = models.resnet50(pretrained=True)

        self.premodel.fc = nn.Identity()
        self.isfreeze = isfreeze
        # param freeze
        if self.isfreeze:
            for param in self.premodel.parameters():
                param.requires_grad = False

        self.premodel_last_dim = list(self.premodel.parameters())[-1].shape[
            0
        ]  # 2048 or 768
        self.output_dim = output_dim
        self.linear1 = nn.Linear(self.premodel_last_dim, self.premodel_last_dim // 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.premodel_last_dim // 2, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = self.premodel(x)  # x:(3,224,224)
        # vitの場合:output.last_hidden_state[:, 0, :]にしたい
        output = self.linear2(self.relu(self.linear1(output)))

        return output


class MultiFrameNet(nn.Module):

    MODAL_NAME = "video"

    def __init__(
        self,
        loss_fct: Callable = None,
        last_activation: Optional[Callable] = nn.Sigmoid(),
        post_loss_process: Optional[Callable] = None,
        unimodal_params_freeze: bool = False,  # resnetのパラメータをフリーズするか
        premodel_type: str = "resnet50",  # 'resnet50' or 'resnet101'
        encoder_dim: int = 256,  # 最終的にマルチモーダルに渡すdim=encoded_stateの次元数
        seriesmodel_type: str = "lstm",  # 'lstm' or 'transformer'
        num_class: int = 1 # 最終アウトプット数
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.class_dim = num_class

        self.seriesmodel_type = seriesmodel_type
        if self.seriesmodel_type == "transformer":

            from src.transformer_model import Transformer

            self.series_dim = self.encoder_dim

            self.seriesmodel = Transformer(
                num_hidden_layers=4, num_attention_heads=4, intermediate_dim=4
            )
            self.to_out = nn.Sequential(
                nn.Linear(self.series_dim, self.class_dim),
            )
        else:

            # LSTMのとき
            self.series_dim = int(self.encoder_dim / 2)
            self.seriesmodel = nn.LSTM(
                self.series_dim, self.series_dim, batch_first=True, bidirectional=True
            )
            self.to_out = nn.Sequential(
                nn.Linear(self.series_dim * 2, 256),
                nn.Dropout(0.2),
                nn.Linear(256, self.class_dim),
            )
            self.attention = Attention(
                input_dim=self.series_dim * 2
            )  # 今はlstmの最初のfn+bnなので2倍

        self.isfreeze = unimodal_params_freeze
        self.premodel_name = premodel_type
        self.unimodel = UniFrameNet(
            premodel_name=self.premodel_name,
            output_dim=self.series_dim,
            isfreeze=self.isfreeze,
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
        self, input_values: torch.Tensor = None, targets: torch.Tensor = None
    ) -> ModelOutput:
        input_values = rearrange(input_values, "b f c h w -> b c h w f")
        outputs = torch.stack(
            [
                self.unimodel(input_values.select(-1, i))
                for i in range(input_values.size(-1))
            ],
            dim=-1,
        )  # (*, 256, 15)
        outputs = rearrange(outputs, "b t f -> b f t")

        if self.seriesmodel_type == "transformer":
            output = self.seriesmodel(outputs)
            last_hidden_state = output.last_hidden_state
            encoded_state = last_hidden_state[:, 0, :]
            attentions = output.attentions[-1]

        else:
            outputs_lstm, _ = self.seriesmodel(outputs)
            # 最初の系列のfh+bh: output[:, 0, :], 最後の系列のfh+bh: output[:, -1, :]は(1, series_dim*2)
            # hidden_stateのfh: output[:, :, :series_dim], hidden_stateのbh: output[:, :, series_dim:]は(1, series_dim*2*15)
            output = outputs_lstm[:, 0, :].view(-1, self.series_dim * 2)

            last_hidden_state = outputs_lstm  # (b, sequence, fh+bh)
            encoded_state = output
            attention_weighted_encoding, attentions = self.attention(last_hidden_state)

        logits = self.last_activation(self.to_out(encoded_state))
        loss = None
        if targets is not None and self.loss_fct is not None:
            loss = self.loss_fct(logits, targets.squeeze_())#.unsqueeze(1).squeeze_()
        logits = self.post_loss_process(logits)

        return ModelOutput(
            encoded_state=encoded_state,
            logits=logits,
            loss=loss,
            last_hidden_state=last_hidden_state,
            attentions=attentions,
        )


class Encoder(MultiFrameNet):
    # 実体はMultiFrameNet

    pass
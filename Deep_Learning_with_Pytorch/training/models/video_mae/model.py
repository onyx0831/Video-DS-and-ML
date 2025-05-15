import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput
from typing import Optional
import logging

from .modeling_videomae import VideoMAEModel


__all__ = ["VideoMAE"]
logger = logging.getLogger(__name__)


class VideoMAE(nn.Module):
    def __init__(
        self,
        model_path: str = "MCG-NJU/videomae-base",
        pretrained: bool = True,
        output_attentions: bool = False,
        use_memory_efficient_attention: bool = False,
    ):
        super().__init__()
        if pretrained:
            logger.info(f"Loading model from {model_path}")
            self.model = VideoMAEModel.from_pretrained(
                model_path,
                use_memory_efficient_attention=use_memory_efficient_attention,
            )
        else:
            # random initialization
            from .configuration_videomae import VideoMAEConfig

            logger.info("VideoMAE is initialized randomly.")
            config = VideoMAEConfig(use_mean_pooling=False)
            self.model = VideoMAEModel(
                config, use_memory_efficient_attention=use_memory_efficient_attention
            )
        self.encoder_dim = self.model.config.hidden_size  # 768
        self.output_attentions = output_attentions
        if output_attentions and use_memory_efficient_attention:
            self.output_attentions = False
            logger.warning(
                "output_attentions is disabled when use_memory_efficient_attention is enabled."
            )

    def forward(
        self, input: torch.Tensor, frame_mask: Optional[torch.BoolTensor] = None, *args
    ) -> ModelOutput:
        """
        args:
            input: video frames shape=(batch_size, frame, 3, H, W)
            frame_mask: mask for padding frames shape=(batch_size, frame)
                        1 for padding frame, 0 for valid frame
        """
        output = self.model(
            input, frame_mask=frame_mask, output_attentions=self.output_attentions
        )
        last_hidden_state = (
            output.last_hidden_state
        )  # (batch_size, seq_len, encoder_dim), seq_len=1568
        # encoded_state is average of last_hidden_state except padding frames
        if output.attention_mask is None:
            encoded_state = last_hidden_state.mean(dim=1)  # (batch_size, encoder_dim)
        else:
            last_hidden_state = last_hidden_state * (
                ~output.attention_mask.unsqueeze(-1)
            ).to(last_hidden_state.dtype)
            encoded_state = last_hidden_state.sum(dim=1) / (
                last_hidden_state.size(1)
                - output.attention_mask.sum(dim=1, keepdim=True)
            )  # (batch_size, encoder_dim)

        if self.output_attentions:
            attentions = (
                output.attentions
            )  # tuple of (batch_size, num_heads, seq_len, seq_len)
        else:
            attentions = None
        return ModelOutput(
            encoded_state=encoded_state,
            last_hidden_state=last_hidden_state,
            attentions=attentions,
        )

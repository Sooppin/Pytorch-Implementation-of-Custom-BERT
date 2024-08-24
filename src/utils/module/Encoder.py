import torch
import torch.nn as nn

from src.utils.attention.MultiHeadAttention import MultiHeadAttention
from src.utils.module.FeedForward import FeedForward


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_size: int = config['model']['hidden_size']
        self.hidden_dropout_prob: float = config['model']['hidden_dropout_prob']
        self.multihead_attention: MultiHeadAttention = MultiHeadAttention(config)
        self.norm1: nn.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.norm2: nn.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.feed_forward: FeedForward = FeedForward(config)
        self.dropout: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor, mask: torch.Tensor = None, training: bool = False) -> torch.Tensor:
        x_norm1: torch.Tensor = self.norm1(hidden_state)
        attention_output: torch.Tensor = self.multihead_attention(x_norm1, x_norm1, x_norm1, mask, training)
        hidden_state: torch.Tensor = attention_output + hidden_state

        x_norm2: torch.Tensor = self.norm2(hidden_state)
        feed_forward_output: torch.Tensor = self.feed_forward(x_norm2, training)
        x_enc: torch.Tensor = feed_forward_output + hidden_state
        if training:
            hidden_state: torch.Tensor = self.dropout(x_enc)

        return hidden_state

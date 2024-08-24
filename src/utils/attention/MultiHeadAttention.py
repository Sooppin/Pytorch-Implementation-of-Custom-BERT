import torch
import torch.nn as nn

from src.utils.attention.AttentionHead import AttentionHead


class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size: int = config['model']['hidden_size']
        self.num_heads: int = config['model']['num_heads']
        self.head_dim: int = config['model']['hidden_size'] // config['model']['num_heads']
        self.attention_heads: nn.ModuleList = nn.ModuleList([AttentionHead(self.hidden_size, self.head_dim) for _ in range(self.num_heads)])
        self.fc: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, training:bool = False) -> torch.Tensor:
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask, training=training) for attention_head in self.attention_heads]
        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state: torch.Tensor = self.fc(hidden_state)
        return hidden_state

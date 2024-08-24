import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):

    def __init__(self, hidden_size, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.query_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, training:bool = False) -> torch.Tensor:
        query: torch.Tensor = self.query_weights(query)
        key: torch.Tensor = self.key_weights(key)
        value: torch.Tensor = self.value_weights(value)

        att_scores: torch.Tensor = torch.matmul(query, key.transpose(1, 2)) / self.head_dim ** 0.5

        if mask is not None:
            mask = mask.to(torch.int)
            att_scores: torch.Tensor = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        att_weights: torch.Tensor = F.softmax(att_scores, dim=-1)
        if training:
            att_weights: torch.Tensor = self.dropout(att_weights)
        n_value: torch.Tensor = torch.matmul(att_weights, value)

        return n_value

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_size: int = config['model']['hidden_size']
        self.intermediate_fc_size: int = self.hidden_size * 4
        self.hidden_dropout_prob: float = config['model']['hidden_dropout_prob']

        self.fc1: nn.Linear = nn.Linear(self.hidden_size, self.intermediate_fc_size)
        self.fc2: nn.Linear = nn.Linear(self.intermediate_fc_size, self.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor, training: bool = False) -> torch.Tensor:
        hidden_state: torch.Tensor = self.fc1(hidden_state)
        hidden_state: torch.Tensor = F.gelu(hidden_state)
        if training:
            hidden_state: torch.Tensor = self.dropout(hidden_state)
        hidden_state: torch.Tensor = self.fc2(hidden_state)

        return hidden_state

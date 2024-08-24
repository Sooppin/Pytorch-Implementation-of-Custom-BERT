import torch
import torch.nn as nn


class PositionalEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.seq_len: int = config['tokenizer']['max_length']
        self.hidden_size: int = config['model']['hidden_size']
        self.positional_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=self.seq_len, embedding_dim=self.hidden_size
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length: int = input_ids.size(1)
        position_ids: torch.Tensor = torch.arange(seq_length, dtype=torch.int32, device=input_ids.device).unsqueeze(0)
        position_embeddings: torch.Tensor = self.positional_embeddings(position_ids)
        return position_embeddings

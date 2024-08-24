import torch
import torch.nn as nn

from src.utils.embedding.PositionalEmbeddings import PositionalEmbeddings


class Embeddings(nn.Module):

    def __init__(self, config, vocab_size):
        super().__init__()

        self.hidden_size: int = config['model']['hidden_size']
        self.vocab_size: int = vocab_size
        self.hidden_dropout_prob: float = config['model']['hidden_dropout_prob']

        self.token_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size
        )
        self.segment_embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=1, embedding_dim=self.hidden_size
        )
        self.positional_embeddings: PositionalEmbeddings = PositionalEmbeddings(config)
        self.dropout: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, training: bool = False) -> torch.Tensor:
        pos_info: torch.Tensor = self.positional_embeddings(input_ids)  # ([1, 512, 768])
        seg_info: torch.Tensor = self.segment_embeddings(segment_ids)
        x: torch.Tensor = self.token_embeddings(input_ids)

        if pos_info.size(0) == 1:
            pos_info = pos_info.expand(x.size(0), -1, -1)
        # print('pos_info', pos_info.size())  # [64, 512, 768])
        # print('seg_info', seg_info.size())  # [64, 512, 768])
        # print('x', x.size())  # ([64, 512, 768])

        x: torch.Tensor = x + pos_info + seg_info
        if training:
            x: torch.Tensor = self.dropout(x)
        return x

    def forward_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids != 0

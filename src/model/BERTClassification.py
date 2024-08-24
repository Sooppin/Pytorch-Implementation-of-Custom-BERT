from typing import Tuple

import torch
import torch.nn as nn

from src.utils.embedding.Embeddings import Embeddings
from src.utils.module.Encoder import Encoder


class BERTClassification(nn.Module):

    def __init__(self, config, vocab_size):
        super(BERTClassification, self).__init__()

        self.num_blocks: int = config['model']['num_blocks']
        self.vocab_size: int = vocab_size
        self.hidden_size: int = config['model']['hidden_size']
        self.num_classes: int = config['trainer']['num_classes']

        self.embed_layer: Embeddings = Embeddings(config, vocab_size)
        self.encoder: nn.ModuleList = nn.ModuleList([Encoder(config) for _ in range(self.num_blocks)])
        self.classifier: nn.Linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input_ids, segment_ids, training=False) -> torch.Tensor:
        x_enc: torch.Tensor = self.embed_layer(input_ids, segment_ids, training)
        mask = self.embed_layer.forward_mask(input_ids)

        for encoder_layer in self.encoder:
            x_enc: torch.Tensor = encoder_layer(x_enc, mask, training=training)

        class_logits: torch.Tensor = self.classifier(x_enc[:, 0, :])

        return class_logits

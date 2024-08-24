import torch
from transformers import AutoTokenizer


class DataTokenizer:

    def __init__(self, tokenizer_path, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    def get_vocab(self):
        return len(self.tokenizer)

    def collate_fn(self, batch):
        texts, label = zip(*batch)
        encodings = self.tokenizer(list(texts),
                                   return_tensors='pt',
                                   max_length=self.max_length,
                                   truncation=True,
                                   padding="max_length")
        encodings['label'] = torch.tensor(label)
        return encodings

import random as rd
import numpy as np
import yaml
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.common.Constants import Constants
from src.dataloader.TextDataLoader import TextDataLoader
from src.model.BERTClassification import BERTClassification
from src.tokenizer.DataTokenizer import DataTokenizer
from src.trainer.BERTTrainer import BERTTrainer


def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def set_seeds(seed, with_cuda):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and with_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run(config):
    # set seed
    set_seeds(config['trainer']['seed'], config['trainer']['with_cuda'])

    # dataset load
    print("Loading Train Dataset...")
    dataset = load_dataset('ag_news')
    train_dataset = TextDataLoader(dataset['train'].select(range(10000)))
    test_dataset = TextDataLoader(dataset['test'].select(range(1000)))

    # create tokenizer and dataloader
    tokenizer = DataTokenizer(config['tokenizer']['tokenizer_path'], config['tokenizer']['max_length'])
    train_loader = DataLoader(train_dataset, batch_size=config['trainer']['batch_size'], shuffle=True, collate_fn=tokenizer.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['trainer']['batch_size'], shuffle=False, collate_fn=tokenizer.collate_fn)

    model = BERTClassification(config, tokenizer.get_vocab())

    # Initialize BERT trainer
    trainer = BERTTrainer(config, model, train_loader, test_loader)

    # train
    trained_model = trainer.train()

    # test
    trainer.test(trained_model)


if __name__ == "__main__":
    config = load_config(Constants.CONFIG_FILE)
    run(config)

import time
from typing import List

from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report


class BERTTrainer:
    def __init__(self, config, model, train_loader, test_loader):
        self.with_cuda: bool = config['trainer']['with_cuda']
        self.cuda_device_id: int = config['trainer']['cuda_device_id']
        self.learning_rate: float = float(config['trainer']['learning_rate'])
        self.epochs: int = config['trainer']['epochs']
        self.save_path: str = config['trainer']['save_path']

        # setup cuda device for BERT training
        cuda_condition = torch.cuda.is_available() and self.with_cuda
        self.device = torch.device(f"cuda:{self.cuda_device_id}" if cuda_condition else "cpu")
        self.model = model.to(self.device)

        print("Total Parameters:", sum(p.nelement() for p in self.model.parameters()))

        # Setting the train and test data loader
        self.train_data = train_loader
        self.test_data = test_loader

        # optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(self.train_data) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss, correct_predictions, total_predictions = 0, 0, 0

            for batch in tqdm(self.train_data, desc=f"EPOCH {epoch+1}/{self.epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                input_ids = batch['input_ids']
                token_type_ids = batch['token_type_ids']
                labels = batch['label']

                logits = self.model(input_ids, token_type_ids, training="train")
                loss = self.criterion(logits, labels)
                predictions = torch.argmax(logits, dim=-1)

                correct_predictions += (predictions == batch['label']).sum().item()
                total_predictions += batch['label'].size(0)
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            avg_train_loss = total_loss / len(self.train_data)
            train_accuracy = correct_predictions / total_predictions
            print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")

        return self.model

    def test(self, model):
        model.eval()
        test_labels, test_preds = [], []
        inference_start_time = time.time()
        verbose = True

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_data, 0), desc="Testing"):
                batch = {k: v.to(self.device) for k, v in batch.itmes()}
                input_ids = batch['input_ids']
                token_type_ids = batch['token_type_ids']
                labels = batch['label']

                logits = model(input_ids, token_type_ids, training="test")
                predictions = torch.argmax(logits, dim=-1)

                test_labels.extend(labels.cpu().numpy())
                test_preds.extend(predictions.cpu().numpy())

                if verbose and (i % 100 == 0):
                    print(f"Batch {i}/{len(self.test_data)} completed")

        inference_end_time = time.time()
        test_inference_duration = inference_end_time - inference_start_time
        test_accuracy = accuracy_score(test_labels, test_preds)
        test_report = classification_report(test_labels, test_preds)

        print(f"Test Accuracy: {test_accuracy}")
        print("Classification Report:\n", test_report)
        print(f"Inference Duration: {test_inference_duration:.2f} seconds")

    def save(self):
        pass

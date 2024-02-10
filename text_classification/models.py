import copy

import torch
from torch import nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class TextClassificationLSTM(nn.Module):

    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self._define_architecture()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embeddings(x)  # [batch_size, max_len, embedding_dim]

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden_state, _) = self.lstm(packed)

        logits = self.linear(hidden_state[-1])

        return logits

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        num_epochs: int = 50,
        patience: int = 5,
    ) -> list[dict]:
        optimizer = Adam(self.parameters(), learning_rate)
        best_score = float("-inf")
        best_model_params = copy.deepcopy(self.state_dict())
        current_patience = 0
        history = []
        for epoch in range(num_epochs):
            self._train_one_epoch(train_loader, optimizer)

            # Evaluate on training and validation set
            train_scores = self._evaluate(train_loader)
            val_scores = self._evaluate(val_loader)
            print(
                f"Epoch {epoch+1:2d}/{num_epochs}: {', '.join([f'{k}={v:.4f}' for k, v in val_scores.items()])}"
            )

            # Check if validation score has improved
            if val_scores["f1_score"] > best_score:
                best_score = val_scores["f1_score"]
                current_patience = 0
                best_model_params = copy.deepcopy(self.state_dict())
            else:
                current_patience += 1
                if current_patience >= patience:
                    print(f"Early stopping - best model from epoch {epoch-current_patience+1}!")
                    break

            # Append scores to history
            train_scores["epoch"] = epoch
            train_scores["set"] = "train"
            history.append(train_scores)
            val_scores["epoch"] = epoch
            val_scores["set"] = "validation"
            history.append(val_scores)

        # Load best model parameters back to the model
        self.load_state_dict(best_model_params)

        return history

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        text_lengths = (inputs != 1).sum(dim=1).cpu()
        logits = self.forward(inputs, text_lengths)
        probs = nn.functional.softmax(logits, dim=1)

        return probs.argmax(dim=1).cpu()

    def save(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        self.load_state_dict(torch.load(file_path))

    def _define_architecture(self) -> None:
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def _train_one_epoch(self, data_loader: DataLoader, optimizer: Optimizer) -> None:
        self.train()
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.type(torch.float).to(self.device)
            text_lengths = (inputs != 1).sum(dim=1).cpu()
            logits = self.forward(inputs, text_lengths)

            loss = nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def _evaluate(self, data_loader: DataLoader) -> dict:
        scores = {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

        self.eval()
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.type(torch.float).to(self.device)
            text_lengths = (inputs != 1).sum(dim=1).cpu()
            logits = self.forward(inputs, text_lengths)
            probs = nn.functional.softmax(logits, dim=1)

            # Calculate scores
            scores["loss"] += nn.functional.cross_entropy(logits, targets).item()
            targets = targets.argmax(dim=1).cpu()
            preds = probs.argmax(dim=1).cpu()
            scores["accuracy"] += float(accuracy_score(targets, preds))
            scores["precision"] += float(precision_score(targets, preds, average="macro"))
            scores["recall"] += float(recall_score(targets, preds, average="macro"))
            scores["f1_score"] += float(f1_score(targets, preds, average="macro"))

        return {key: value / len(data_loader) for key, value in scores.items()}

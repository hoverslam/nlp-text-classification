import copy
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer
from tokenizers import Tokenizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm.notebook import tqdm


class BaseClassifier(ABC, nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def forward(self) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self) -> torch.Tensor:
        pass

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 3e-4,
        num_epochs: int = 50,
        patience: int = 5,
        verbose: int = 1,
    ) -> list[dict]:
        optimizer = AdamW(self.parameters(), learning_rate)
        best_score = float("-inf")
        best_model_params = copy.deepcopy(self.state_dict())
        current_patience = 0
        history = []
        for epoch in range(num_epochs):
            self._train_one_epoch(train_loader, optimizer, verbose)

            # Evaluate on training and validation set
            train_scores = self.evaluate(train_loader, verbose)
            val_scores = self.evaluate(val_loader, verbose)
            if verbose > 0:
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
                    if verbose > 0:
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
    def evaluate(self, data_loader: DataLoader, verbose: int) -> dict:
        scores = {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

        self.eval()
        for batch in tqdm(data_loader, desc="Evaluation", disable=(verbose < 2)):
            input_ids, attention_mask, labels, _ = batch
            logits = self(input_ids.to(self.device), attention_mask.to(self.device))
            probs = nn.functional.softmax(logits, dim=1)

            # Calculate scores
            scores["loss"] += nn.functional.cross_entropy(logits, labels.to(self.device)).item()
            labels = labels.argmax(dim=1).cpu()
            preds = probs.argmax(dim=1).cpu()
            scores["accuracy"] += float(accuracy_score(labels, preds))
            scores["precision"] += float(precision_score(labels, preds, average="macro"))
            scores["recall"] += float(recall_score(labels, preds, average="macro"))
            scores["f1_score"] += float(f1_score(labels, preds, average="macro"))

        return {key: value / len(data_loader) for key, value in scores.items()}

    def save(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        self.load_state_dict(torch.load(file_path, map_location=self.device))

    def _train_one_epoch(self, data_loader: DataLoader, optimizer: Optimizer, verbose: int) -> None:
        self.train()
        for batch in tqdm(data_loader, desc="Training", disable=(verbose < 2)):
            input_ids, attention_masks, labels, _ = batch
            logits = self(input_ids.to(self.device), attention_masks.to(self.device))
            loss = nn.functional.cross_entropy(logits, labels.to(self.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class LSTMClassifier(BaseClassifier):
    """A LSTM model for text classification."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 100,
        hidden_size: int = 300,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = 30_000
        self._define_architecture()
        self.to(self.device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # If all tokens in a row are "padding" it cannot be packed since the lengths is 0.
        # Set input_ids to zeros and attention_mask to ones.
        mask = torch.all(input_ids == 1, dim=1)
        input_ids[mask] = 0
        attention_mask[mask] = 1

        embedded = self.embeddings(input_ids)  # [batch_size, seq_len, embedding_dim]

        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden_state, _) = self.lstm(packed)

        logits = self.linear(hidden_state[-1])

        return logits

    @torch.no_grad()
    def predict(self, sentences: list[str]) -> list[int]:
        self.eval()
        tokenizer = Tokenizer.from_file("./models/lstm_tokenizer.json")
        tokenizer.enable_padding(pad_id=1, pad_token="[PAD]", length=512)
        tokenizer.enable_truncation(max_length=512)
        encoded_inputs = tokenizer.encode_batch(sentences)
        input_ids = [x.ids for x in encoded_inputs]
        attention_mask = [x.attention_mask for x in encoded_inputs]
        input_ids = torch.tensor(input_ids, dtype=torch.int).to(self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int).to(self.device)
        logits = self(input_ids, attention_mask)
        probs = nn.functional.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        return preds.tolist()

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


class DistilBERTClassifier(BaseClassifier):
    """A fine-tuned DistilBERT for text classification."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self._define_architecture(num_classes)
        self.to(self.device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.bert(input_ids, attention_mask)[0]  # [batch_size, seq_len, dim]
        cls_token = hidden_states[:, 0, :]  # [batch_size, dim]
        logits = self.classifier(cls_token)

        return logits

    @torch.no_grad()
    def predict(self, sentences: list[str]) -> list[int]:
        self.eval()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        encoded_inputs = tokenizer(sentences, padding="max_length", truncation=True)
        input_ids = torch.tensor(encoded_inputs["input_ids"], dtype=torch.int).to(self.device)
        attention_mask = torch.tensor(encoded_inputs["attention_mask"], dtype=torch.int).to(
            self.device
        )
        logits = self(input_ids, attention_mask)
        probs = nn.functional.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        return preds.tolist()

    def _define_architecture(self, num_classes: int) -> None:
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float32)
        self.classifier = nn.Linear(768, num_classes)

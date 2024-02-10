import datasets
from tokenizers import Tokenizer

import torch
from torch.utils.data import Dataset


class SpamDetectionDataset(Dataset):

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._sentences, self._labels = self._load_dataset()
        self._class_names = ["no_spam", "spam"]

    def __len__(self) -> int:
        return len(self._sentences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = self._sentences[idx]
        encoded_text = self._tokenizer.encode(text).ids
        encoded_text = torch.tensor(encoded_text, dtype=torch.int)

        label = torch.zeros(len(self._class_names), dtype=torch.int8)
        label[self._class_names.index(self._labels[idx])] = 1

        # Fallback if the encoded text is fully padded
        if torch.all(encoded_text == 1):
            encoded_text = torch.zeros_like(encoded_text)
            label = torch.tensor([1, 0], dtype=torch.int8)

        return encoded_text, label

    def get_class_names(self, predictions: torch.Tensor) -> list:
        return [self._class_names[p] for p in predictions]

    def _load_dataset(self) -> tuple[list[str], list[str]]:
        dataset = datasets.load_dataset("Deysi/spam-detection-dataset")
        dataset = datasets.concatenate_datasets([dataset["train"], dataset["test"]])
        sentences, labels = [], []
        for sample in dataset:
            if len(sample["text"]) > 0:
                sentences.append(sample["text"])
                labels.append(sample["label"])

        return sentences, labels


class AllScamSpamDataset(Dataset):

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._sentences, self._labels = self._load_dataset()
        self._class_names = ["no_spam", "spam"]

    def __len__(self) -> int:
        return len(self._sentences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = self._sentences[idx]
        encoded_text = self._tokenizer.encode(text).ids
        encoded_text = torch.tensor(encoded_text, dtype=torch.int)

        label = torch.zeros(len(self._class_names), dtype=torch.int8)
        label[self._labels[idx]] = 1

        # Fallback if the encoded text is fully padded
        if torch.all(encoded_text == 1):
            encoded_text = torch.zeros_like(encoded_text)
            label = torch.tensor([1, 0], dtype=torch.int8)

        return encoded_text, label

    def get_class_names(self, predictions: torch.Tensor) -> list:
        return [self._class_names[p] for p in predictions]

    def _load_dataset(self) -> tuple[list[str], list[int]]:
        dataset = datasets.load_dataset("FredZhang7/all-scam-spam")
        dataset = datasets.concatenate_datasets([dataset["train"]])
        sentences, labels = [], []
        for sample in dataset:
            if len(sample["text"]) > 0:
                sentences.append(sample["text"])
                labels.append(sample["is_spam"])

        return sentences, labels


class EmotionsDataset(Dataset):

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._sentences, self._labels = self._load_dataset()
        self._class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def __len__(self) -> int:
        return len(self._sentences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        text = self._sentences[idx]
        encoded_text = self._tokenizer.encode(text).ids
        encoded_text = torch.tensor(encoded_text, dtype=torch.int)

        label = torch.zeros(len(self._class_names), dtype=torch.int8)
        label[self._labels[idx]] = 1

        # Fallback if the encoded text is fully padded
        if torch.all(encoded_text == 1):
            encoded_text = torch.zeros_like(encoded_text)
            label = torch.tensor([1, 0], dtype=torch.int8)

        return encoded_text, label

    def get_class_names(self, predictions: torch.Tensor) -> list:
        return [self._class_names[p] for p in predictions]

    def _load_dataset(self) -> tuple[list[str], list[int]]:
        dataset = datasets.load_dataset("jeffnyman/emotions", trust_remote_code=True)
        dataset = datasets.concatenate_datasets(
            [dataset["train"], dataset["validation"], dataset["test"]]
        )
        sentences, labels = [], []
        for sample in dataset:
            if len(sample["text"]) > 0:
                sentences.append(sample["text"])
                labels.append(sample["label"])

        return sentences, labels

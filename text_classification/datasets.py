import warnings, os

import datasets
from tokenizers import Tokenizer
from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class SpamDetectionDataset(Dataset):
    """A PyTorch Dataset for a spam detection task.
    https://huggingface.co/datasets/FredZhang7/all-scam-spam
    """

    def __init__(self, model: str) -> None:
        """Initialize the Spam Detection dataset.

        Args:
            model (str): The name of the model to use for tokenization.
        """
        super().__init__()
        self._input_ids, self._attention_mask, self._labels, self._text = self._load_dataset(model)
        self._class_names = ["no_spam", "spam"]

    @property
    def class_names(self) -> list[str]:
        """Return the class names.

        Returns:
            int: The class names of the dataset.
        """
        return self._class_names

    @property
    def num_classes(self) -> int:
        """Return the number of classes.

        Returns:
            int: The number of classes in the dataset.
        """
        return len(self._class_names)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Retrieve a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing input_ids, attention_mask, label, and the original text.
        """
        input_ids = torch.tensor(self._input_ids[idx], dtype=torch.int)
        attention_mask = torch.tensor(self._attention_mask[idx], dtype=torch.int)

        label = torch.zeros(len(self._class_names), dtype=torch.float32)
        label[self._labels[idx]] = 1

        return input_ids, attention_mask, label, self._text[idx]

    def get_class_names(self, predictions: torch.Tensor) -> list:
        return [self._class_names[p] for p in predictions]

    def _load_dataset(self, model: str) -> tuple[list[int], list[int], list[int], list[str]]:
        """Load and preprocesse the dataset.

        Args:
            model (str): The name of the model to use for tokenization.

        Returns:
            tuple: A tuple containing input_ids, attention_mask, labels, and the original text.
        """
        dataset = datasets.load_dataset("FredZhang7/all-scam-spam")
        dataset = datasets.concatenate_datasets([dataset["train"]])

        match model:
            case "distilbert":
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                text, labels = zip(*[(sample["text"], sample["is_spam"]) for sample in dataset])
                encoded_inputs = tokenizer(text, padding="max_length", truncation=True)
                input_ids = encoded_inputs["input_ids"]
                attention_mask = encoded_inputs["attention_mask"]
            case "lstm":
                tokenizer = Tokenizer.from_file(f"./models/lstm_tokenizer.json")
                tokenizer.enable_padding(pad_id=1, pad_token="[PAD]", length=512)
                tokenizer.enable_truncation(max_length=512)
                text, labels = zip(*[(sample["text"], sample["is_spam"]) for sample in dataset])
                encoded_inputs = tokenizer.encode_batch(text)
                input_ids = [x.ids for x in encoded_inputs]
                attention_mask = [x.attention_mask for x in encoded_inputs]
            case _:
                raise KeyError(f"Tokenizer for {model} is not available.")

        return list(input_ids), list(attention_mask), labels, text


class EmotionsDataset(Dataset):
    """A PyTorch Dataset for a text classification dataset about emotions.
    https://huggingface.co/datasets/jeffnyman/emotions
    """

    def __init__(self, model: str) -> None:
        """Initialize the Emotions dataset.

        Args:
            model (str): The name of the model to use for tokenization.
        """
        super().__init__()
        self._input_ids, self._attention_mask, self._labels, self._text = self._load_dataset(model)
        self._class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    @property
    def class_names(self) -> list[str]:
        """Return the class names.

        Returns:
            int: The class names of the dataset.
        """
        return self._class_names

    @property
    def num_classes(self) -> int:
        """Return the number of classes.

        Returns:
            int: The number of classes in the dataset.
        """
        return len(self._class_names)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Retrieve a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing input_ids, attention_mask, label, and the original text.
        """
        input_ids = torch.tensor(self._input_ids[idx], dtype=torch.int)
        attention_mask = torch.tensor(self._attention_mask[idx], dtype=torch.int)

        label = torch.zeros(len(self._class_names), dtype=torch.float32)
        label[self._labels[idx]] = 1

        return input_ids, attention_mask, label, self._text[idx]

    def get_class_names(self, predictions: torch.Tensor) -> list:
        """Convert prediction indices to emotion class names.

        Args:
            predictions (torch.Tensor): The tensor containing prediction indices.

        Returns:
            list: A list of emotion class names corresponding to the prediction indices.
        """
        return [self._class_names[p] for p in predictions]

    def _load_dataset(self, model: str) -> tuple[list[int], list[int], list[int], list[str]]:
        """Load and preprocesse the Emotions dataset.

        Args:
            model (str): The name of the model to use for tokenization.

        Returns:
            tuple: A tuple containing input_ids, attention_mask, labels, and the original text.
        """
        dataset = datasets.load_dataset("jeffnyman/emotions", trust_remote_code=True)
        dataset = datasets.concatenate_datasets(
            [dataset["train"], dataset["validation"], dataset["test"]]
        )

        match model:
            case "distilbert":
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                text, labels = zip(*[(sample["text"], sample["label"]) for sample in dataset])
                encoded_inputs = tokenizer(text, padding="max_length", truncation=True)
                input_ids = encoded_inputs["input_ids"]
                attention_mask = encoded_inputs["attention_mask"]
            case "lstm":
                tokenizer = Tokenizer.from_file("./models/lstm_tokenizer.json")
                tokenizer.enable_padding(pad_id=1, pad_token="[PAD]", length=512)
                tokenizer.enable_truncation(max_length=512)
                text, labels = zip(*[(sample["text"], sample["label"]) for sample in dataset])
                encoded_inputs = tokenizer.encode_batch(text)
                input_ids = [x.ids for x in encoded_inputs]
                attention_mask = [x.attention_mask for x in encoded_inputs]
            case _:
                raise KeyError(f"Tokenizer for {model} is not available.")

        return list(input_ids), list(attention_mask), labels, text

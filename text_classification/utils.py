from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import WhitespaceSplit, Punctuation, Digits
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset


def train_lstm_tokenizer() -> None:
    # Define tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([WhitespaceSplit(), Punctuation(), Digits()])

    # Train tokenizer on wikitext
    wikitext = load_dataset("wikitext", "wikitext-103-raw-v1")
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"])
    tokenizer.train_from_iterator([t["text"] for t in wikitext["train"]], trainer)

    # Save tokenizer
    tokenizer.save("./models/lstm_tokenizer.json")


import torch
from pathlib import Path

from tokenizer.char_tokenizer import CharTokenizer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "input.txt"
TRAIN_FILE = PROJECT_ROOT / "data" / "train.pt"
VAL_FILE = PROJECT_ROOT / "data" / "val.pt"
VOCAB_FILE = PROJECT_ROOT / "tokenizer" / "vocab.json"


def prepare_data():

    print("Loading dataset...")

    # Read original text
    text = INPUT_FILE.read_text(encoding="utf-8")

    if len(text) < 100:
        raise ValueError("Dataset is too small")

    print("Total characters:", len(text))

    # Split text before building vocabulary
    split_index = int(len(text) * 0.90)

    train_text = text[:split_index]
    val_text = text[split_index:]

    print("Training characters:", len(train_text))
    print("Validation characters:", len(val_text))

    # Build tokenizer using training text only
    tokenizer = CharTokenizer(text=train_text)

    print("Vocabulary size:", len(tokenizer.vocab))

    # Save vocabulary for future inference
    tokenizer.save(VOCAB_FILE)

    # Convert text into token IDs
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    # Count unknown characters in validation data
    unk_id = tokenizer.char_to_id[tokenizer.unk_token]

    unknown_count = val_ids.count(unk_id)

    print("Unknown validation characters:", unknown_count)

    # Convert token IDs into PyTorch tensors
    train_tensor = torch.tensor(
        train_ids,
        dtype=torch.long
    )

    val_tensor = torch.tensor(
        val_ids,
        dtype=torch.long
    )

    # Save encoded datasets
    torch.save(train_tensor, TRAIN_FILE)
    torch.save(val_tensor, VAL_FILE)

    print("\nData preparation completed!")
    print("Training tensor:", train_tensor.shape)
    print("Validation tensor:", val_tensor.shape)
    print("Vocabulary saved:", VOCAB_FILE)
    print("Training data saved:", TRAIN_FILE)
    print("Validation data saved:", VAL_FILE)


if __name__ == "__main__":
    prepare_data()


from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

TRAIN_TEXT = ROOT / "data/cleaned/training_clean.txt"
VAL_TEXT = ROOT / "data/cleaned/validation_clean.txt"

TOKENIZER_FILE = ROOT / "tokenizer/cipai_clean_bpe.json"

TRAIN_OUTPUT = ROOT / "data/train_clean_bpe.pt"
VAL_OUTPUT = ROOT / "data/val_clean_bpe.pt"


def encode_file(path, tokenizer):
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")

    encoded = tokenizer.encode(text)
    token_ids = torch.tensor(encoded.ids, dtype=torch.long)

    print(f"\n{path.name}")
    print("Characters:", f"{len(text):,}")
    print("Tokens:", f"{len(token_ids):,}")

    return token_ids


def main():
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            "Clean tokenizer missing. Train it first."
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    print("Vocabulary size:", tokenizer.get_vocab_size())

    train_data = encode_file(TRAIN_TEXT, tokenizer)
    val_data = encode_file(VAL_TEXT, tokenizer)

    torch.save(train_data, TRAIN_OUTPUT)
    torch.save(val_data, VAL_OUTPUT)

    print("\nData preparation complete.")
    print("Training tensor:", TRAIN_OUTPUT)
    print("Validation tensor:", VAL_OUTPUT)


if __name__ == "__main__":
    main()

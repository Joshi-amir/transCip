
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_FILE = ROOT / "tokenizer/cipai_clean_bpe.json"

TRAIN_FILE = ROOT / "data/train_clean_bpe.pt"
VAL_FILE = ROOT / "data/val_clean_bpe.pt"

BLOCK_SIZE = 512


def validate_tensor(name, path, vocab_size):
    if not path.exists():
        raise FileNotFoundError(path)

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    assert data.ndim == 1, f"{name}: Tensor must be one-dimensional."
    assert data.dtype == torch.long, f"{name}: Must use torch.long."
    assert len(data) > BLOCK_SIZE + 1, f"{name}: Dataset too small."

    minimum = int(data.min())
    maximum = int(data.max())

    assert minimum >= 0, f"{name}: Negative token ID."
    assert maximum < vocab_size, f"{name}: Token ID outside vocabulary."

    print(f"\n{name}: PASSED")
    print("Tokens:", f"{len(data):,}")
    print("Unique IDs:", len(torch.unique(data)))
    print("Minimum ID:", minimum)
    print("Maximum ID:", maximum)

    return data


def main():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    vocab_size = tokenizer.get_vocab_size()

    print("Vocabulary size:", vocab_size)

    train_data = validate_tensor(
        "Training data", TRAIN_FILE, vocab_size
    )

    val_data = validate_tensor(
        "Validation data", VAL_FILE, vocab_size
    )

    print("\nBoth token datasets passed validation.")
    print("Training tokens:", f"{len(train_data):,}")
    print("Validation tokens:", f"{len(val_data):,}")


if __name__ == "__main__":
    main()

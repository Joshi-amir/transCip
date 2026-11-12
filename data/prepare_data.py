
import torch
from pathlib import Path
from tokenizers import Tokenizer


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_TEXT = PROJECT_ROOT / "data" / "input.txt"
VAL_TEXT = PROJECT_ROOT / "data" / "wikitext_validation.txt"

TOKENIZER_FILE = PROJECT_ROOT / "tokenizer" / "cipai_bpe.json"

TRAIN_OUTPUT = PROJECT_ROOT / "data" / "train_bpe.pt"
VAL_OUTPUT = PROJECT_ROOT / "data" / "val_bpe.pt"


# ==========================================
# LOAD TOKENIZER
# ==========================================

def load_tokenizer():

    print("Loading BPE tokenizer...")

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError("BPE tokenizer not found.")

    return Tokenizer.from_file(str(TOKENIZER_FILE))


# ==========================================
# ENCODE TEXT
# ==========================================

def encode_file(path, tokenizer):

    print(f"\nReading: {path.name}")

    text = path.read_text(encoding="utf-8")

    print("Characters:", len(text))

    encoded = tokenizer.encode(text)

    token_ids = torch.tensor(
        encoded.ids,
        dtype=torch.long
    )

    print("Tokens:", len(token_ids))

    return token_ids


# ==========================================
# PREPARE DATASET
# ==========================================

def prepare_data():

    print("Preparing cipAI BPE datasets...")

    tokenizer = load_tokenizer()

    train_data = encode_file(
        TRAIN_TEXT,
        tokenizer
    )

    val_data = encode_file(
        VAL_TEXT,
        tokenizer
    )

    if len(train_data) <= 512 or len(val_data) <= 512:
        raise ValueError("Dataset is too small.")

    torch.save(train_data, TRAIN_OUTPUT)
    torch.save(val_data, VAL_OUTPUT)

    print("\nDataset preparation completed!")

    print("BPE vocabulary:", tokenizer.get_vocab_size())
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))

    print("\nSaved files:")
    print(TRAIN_OUTPUT)
    print(VAL_OUTPUT)


if __name__ == "__main__":
    prepare_data()

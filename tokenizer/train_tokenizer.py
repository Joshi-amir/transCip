
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "input.txt"

OUTPUT_FILE = PROJECT_ROOT / "tokenizer" / "cipai_bpe.json"


# ==========================================
# CONFIGURATION
# ==========================================

VOCAB_SIZE = 8000
MIN_FREQUENCY = 2


# ==========================================
# LOAD TRAINING TEXT
# ==========================================

def read_training_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_FILE}"
        )

    print("Loading WikiText training data...")

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        text = file.read()

    print("Characters loaded:", len(text))

    return text


# ==========================================
# TRAIN BPE TOKENIZER
# ==========================================

def train_tokenizer():

    text = read_training_data()

    print("\nInitializing BPE tokenizer...")

    tokenizer = Tokenizer(
        BPE(unk_token="[UNK]")
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=["[UNK]"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )

    print("\nTraining BPE tokenizer...")

    tokenizer.train_from_iterator(
        iter(text.splitlines()),
        trainer=trainer
    )

    # ======================================
    # SAVE TOKENIZER
    # ======================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer.save(str(OUTPUT_FILE))

    print("\nTokenizer training completed!")

    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("Saved at:", OUTPUT_FILE)

    # ======================================
    # TEST TOKENIZER
    # ======================================

    sample = "Artificial intelligence is fascinating."

    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)

    print("\nTokenizer test:")
    print("Original:", sample)
    print("Token IDs:", encoded.ids)
    print("Number of tokens:", len(encoded.ids))
    print("Decoded:", decoded)

    assert decoded == sample, "Tokenizer round-trip test failed!"

    print("\nTokenizer test passed!")


if __name__ == "__main__":
    train_tokenizer()

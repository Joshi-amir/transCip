
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data/cleaned/training_clean.txt"
OUTPUT_FILE = ROOT / "tokenizer/cipai_clean_bpe.json"

VOCAB_SIZE = 8000
MIN_FREQUENCY = 2


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Cleaned training data missing. Run data.clean_dataset first."
        )

    text = DATA_FILE.read_text(encoding="utf-8")

    print("Training clean BPE tokenizer...")
    print("Training characters:", f"{len(text):,}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=["[UNK]"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        iter(text.splitlines()),
        trainer=trainer,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT_FILE))

    print("\nTokenizer saved:", OUTPUT_FILE)
    print("Vocabulary size:", tokenizer.get_vocab_size())

    # Round-trip test
    sample = "Artificial intelligence is fascinating."

    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)

    print("\nTokenizer test:")
    print("Original:", sample)
    print("Decoded:", decoded)

    assert decoded == sample, "Tokenizer round-trip test failed!"

    print("Tokenizer test passed.")


if __name__ == "__main__":
    main()

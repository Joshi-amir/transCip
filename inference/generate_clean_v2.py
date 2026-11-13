
import argparse
import torch

from pathlib import Path
from tokenizers import Tokenizer

from models.gpt import GPTLanguageModel

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = ROOT / "models/checkpoints/cipai_clean_v2.pt"
TOKENIZER_FILE = ROOT / "tokenizer/cipai_clean_bpe.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    config = checkpoint["model_config"]

    model = GPTLanguageModel(**config).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def generate(prompt, max_new_tokens=150):
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    model, checkpoint = load_model()

    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )

    generated_ids = output[0].tolist()

    print("\n===== cipAI Clean V2 =====")
    print("Checkpoint step:", checkpoint["step"])
    print("Validation loss:", checkpoint["val_loss"])
    print("\nGenerated text:\n")

    print(tokenizer.decode(generated_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="Artificial intelligence",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=150,
    )

    args = parser.parse_args()

    generate(args.prompt, args.tokens)


if __name__ == "__main__":
    main()

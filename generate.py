
import torch
from pathlib import Path
from tokenizers import Tokenizer

from models.gpt import GPTLanguageModel


# Paths and device
PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_FILE = (
    PROJECT_ROOT / "models" / "checkpoints" / "cipai_clean_v2.pt"
)

TOKENIZER_FILE = (
    PROJECT_ROOT / "tokenizer" / "cipai_clean_bpe.json"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_FILE}"
        )

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_FILE}"
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=True
    )

    config = checkpoint["model_config"]

    model = GPTLanguageModel(**config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, config


@torch.inference_mode()
def generate_text(
    prompt,
    model,
    tokenizer,
    config,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50
):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    encoded = tokenizer.encode(prompt)

    if not encoded.ids:
        raise ValueError("Please enter a valid prompt.")

    idx = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(max_new_tokens):
        idx_context = idx[:, -config["context_length"]:]

        logits, _ = model(idx_context)

        logits = logits[:, -1, :] / temperature

        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)

        logits[logits < values[:, [-1]]] = float("-inf")

        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        idx = torch.cat((idx, next_token), dim=1)

    return tokenizer.decode(idx[0].tolist())


def main():
    model, tokenizer, config = load_model()

    parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("cipAI Clean V2 loaded successfully!")
    print("Device:", DEVICE)
    print("Parameters:", f"{parameters:,}")
    print("Vocabulary:", tokenizer.get_vocab_size())

    print("\nType 'exit' to quit.")

    while True:
        prompt = input("\nYou: ").strip()

        if prompt.lower() == "exit":
            print("Closing cipAI.")
            break

        if not prompt:
            continue

        try:
            output = generate_text(
                prompt,
                model,
                tokenizer,
                config
            )

            print("\ncipAI:", output)

        except Exception as error:
            print("Generation error:", error)


if __name__ == "__main__":
    main()

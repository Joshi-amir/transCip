import torch
import json
from pathlib import Path
import argparse
import sys

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.gpt import GPTLanguageModel

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
BEST_MODEL_FILE = CHECKPOINT_DIR / "best_model.pt"
VOCAB_FILE = PROJECT_ROOT / "tokenizer" / "vocab.json"

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_text(prompt_text, max_new_tokens=500):
    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(f"Checkpoint not found at {BEST_MODEL_FILE}. Run training first.")
    
    if not VOCAB_FILE.exists():
        raise FileNotFoundError(f"Vocab file not found at {VOCAB_FILE}.")

    # Load Vocabulary
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Character-level mappings (temporary until BPE upgrade)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: "".join([itos.get(i, "") for i in l])

    # Load Checkpoint
    checkpoint = torch.load(BEST_MODEL_FILE, map_location=device, weights_only=True)
    config = checkpoint["model_config"]

    # Initialize model with saved config
    model = GPTLanguageModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        context_length=config["context_length"],
        dropout=config["dropout"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Encode prompt
    context_tokens = encode(prompt_text)
    if not context_tokens:
        context_tokens = [0] # Fallback if empty prompt

    x = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)

    print("\n" + "=" * 55)
    print(f"PROMPT: {prompt_text}")
    print("=" * 55)

    # Generate
    with torch.no_grad():
        generated_tokens = model.generate(x, max_new_tokens=max_new_tokens)[0].tolist()

    output_text = decode(generated_tokens)
    print(output_text)
    print("=" * 55)
    print(f"Checkpoint step: {checkpoint.get('step', 'N/A')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"Device: {device}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using trained cipAI model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="\n",
        help="Starting text prompt for the model."
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=500,
        help="Number of new tokens to generate."
    )

    args = parser.parse_args()
    generate_text(prompt_text=args.prompt, max_new_tokens=args.tokens)
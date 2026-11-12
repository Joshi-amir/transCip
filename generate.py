
import torch
from pathlib import Path
from tokenizers import Tokenizer

from models.gpt import GPTLanguageModel


# ==========================================
# PATHS AND DEVICE
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_FILE = (
    PROJECT_ROOT / "models" / "checkpoints" / "cipai_v2_bpe.pt"
)

TOKENIZER_FILE = (
    PROJECT_ROOT / "tokenizer" / "cipai_bpe.json"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# LOAD TOKENIZER
# ==========================================

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))


# ==========================================
# LOAD MODEL
# ==========================================

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=True
)

config = checkpoint["model_config"]

model = GPTLanguageModel(**config).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("cipAI v2 loaded successfully!")
print("Device:", DEVICE)
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ==========================================
# GENERATE TEXT
# ==========================================

@torch.inference_mode()
def generate_text(
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50
):

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

        # Keep only the top-k most likely tokens.
        values, _ = torch.topk(logits, top_k)

        logits[logits < values[:, [-1]]] = float("-inf")

        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        idx = torch.cat((idx, next_token), dim=1)

    return tokenizer.decode(idx[0].tolist())


# ==========================================
# INTERACTIVE LOOP
# ==========================================

print("\nType 'exit' to quit.")

while True:

    prompt = input("\nYou: ").strip()

    if prompt.lower() == "exit":
        print("Closing cipAI.")
        break

    if not prompt:
        continue

    try:
        output = generate_text(prompt)

        print("\ncipAI:", output)

    except Exception as error:
        print("Generation error:", error)

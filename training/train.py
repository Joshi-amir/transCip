import torch
import torch.nn as nn
from pathlib import Path
import json
import sys 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # <-- Add this line so Python finds 'models'

from models.gpt import GPTLanguageModel
from torch.amp import autocast, GradScaler


# ==========================================
# CONFIGURATION (~40-50M Parameter Scale)
# ==========================================



TRAIN_FILE = PROJECT_ROOT / "data" / "train.pt"
VAL_FILE = PROJECT_ROOT / "data" / "val.pt"
VOCAB_FILE = PROJECT_ROOT / "tokenizer" / "vocab.json"

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
BEST_MODEL_FILE = CHECKPOINT_DIR / "best_model.pt"

# Model configuration (Scaled up for RTX 4060 8GB VRAM)
EMBEDDING_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 12
CONTEXT_LENGTH = 512
DROPOUT = 0.1

# Training configuration
BATCH_SIZE = 16
BLOCK_SIZE = 512
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 50

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

SEED = 1337


# ==========================================
# DEVICE AND REPRODUCIBILITY
# ==========================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Training device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError(
            "Training data missing. Run data.prepare_data first."
        )

    train_data = torch.load(
        TRAIN_FILE,
        map_location="cpu",
        weights_only=True
    )

    val_data = torch.load(
        VAL_FILE,
        map_location="cpu",
        weights_only=True
    )

    if len(train_data) <= BLOCK_SIZE:
        raise ValueError("Training dataset is too small")

    if len(val_data) <= BLOCK_SIZE:
        raise ValueError("Validation dataset is too small")

    return train_data, val_data


# ==========================================
# CREATE RANDOM TRAINING BATCHES
# ==========================================

def get_batch(data):

    indices = torch.randint(
        0,
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        data[i:i + BLOCK_SIZE]
        for i in indices
    ])

    y = torch.stack([
        data[i + 1:i + BLOCK_SIZE + 1]
        for i in indices
    ])

    return x.to(device), y.to(device)


# ==========================================
# VALIDATION
# ==========================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data):

    model.eval()
    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = torch.zeros(EVAL_ITERS)

        for i in range(EVAL_ITERS):
            x, y = get_batch(data)
            with autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[i] = loss.item()

        results[name] = losses.mean().item()

    model.train()
    return results


# ==========================================
# TRAINING
# ==========================================

def train():

    train_data, val_data = load_data()

    with open(VOCAB_FILE, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    vocab_size = len(vocab)

    print("Vocabulary size:", vocab_size)
    print("Training characters:", len(train_data))
    print("Validation characters:", len(val_data))

    # Create model (~40-50M parameters)
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        context_length=CONTEXT_LENGTH,
        dropout=DROPOUT
    ).to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Total parameters: {total_parameters:,} (~{total_parameters / 1e6:.1f}M)")

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Initialize GradScaler for mixed precision (bfloat16)
    scaler = GradScaler('cuda')

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")

    initial = estimate_loss(
        model,
        train_data,
        val_data
    )

    print("\nInitial losses:")
    print(f"Train: {initial['train']:.4f}")
    print(f"Validation: {initial['val']:.4f}")

    # Main training loop
    for step in range(MAX_ITERS):

        model.train()
        x, y = get_batch(train_data)

        # Mixed precision forward pass
        with autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became invalid")

        optimizer.zero_grad(set_to_none=True)

        # Scaled backpropagation
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        scaler.step(optimizer)
        scaler.update()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:

            losses = estimate_loss(
                model,
                train_data,
                val_data
            )

            print(
                f"Step {step:4d} | "
                f"Train loss: {losses['train']:.4f} | "
                f"Val loss: {losses['val']:.4f}"
            )

            if losses["val"] < best_val_loss:

                best_val_loss = losses["val"]

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "vocab_size": vocab_size,
                        "embedding_dim": EMBEDDING_DIM,
                        "num_heads": NUM_HEADS,
                        "num_layers": NUM_LAYERS,
                        "context_length": CONTEXT_LENGTH,
                        "dropout": DROPOUT
                    },
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"]
                }

                torch.save(
                    checkpoint,
                    BEST_MODEL_FILE
                )

                print("Best model checkpoint saved!")

    print("\nTraining completed!")
    print("Best validation loss:", round(best_val_loss, 4))
    print("Model saved at:", BEST_MODEL_FILE)

    if device == "cuda":
        print(
            "Peak GPU memory:",
            round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3
            ),
            "GB"
        )


if __name__ == "__main__":
    train()
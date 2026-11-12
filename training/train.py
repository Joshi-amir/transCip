
import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer
from contextlib import nullcontext

from models.gpt import GPTLanguageModel


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "data" / "train_bpe.pt"
VAL_FILE = PROJECT_ROOT / "data" / "val_bpe.pt"

TOKENIZER_FILE = PROJECT_ROOT / "tokenizer" / "cipai_bpe.json"

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
BEST_MODEL_FILE = CHECKPOINT_DIR / "cipai_v2_bpe.pt"

EMBEDDING_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 12
CONTEXT_LENGTH = 512
DROPOUT = 0.1

BATCH_SIZE = 4
BLOCK_SIZE = 512

MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 20

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

SEED = 1337


# ==========================================
# DEVICE
# ==========================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

USE_BF16 = (
    device == "cuda"
    and torch.cuda.is_bf16_supported()
)

def autocast_context():
    if USE_BF16:
        return torch.autocast("cuda", dtype=torch.bfloat16)

    return nullcontext()


print("Training device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16:", USE_BF16)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        raise FileNotFoundError(
            "BPE datasets missing. Run data/prepare_data.py first."
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
        raise ValueError("Training dataset too small.")

    if len(val_data) <= BLOCK_SIZE:
        raise ValueError("Validation dataset too small.")

    return train_data, val_data


# ==========================================
# CREATE BATCHES
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

            with autocast_context():
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

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    vocab_size = tokenizer.get_vocab_size()

    print("Vocabulary size:", vocab_size)
    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        context_length=CONTEXT_LENGTH,
        dropout=DROPOUT
    ).to(device)

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(
        f"Total parameters: {total_parameters:,} "
        f"(~{total_parameters / 1e6:.1f}M)"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    initial = estimate_loss(
        model,
        train_data,
        val_data
    )

    print("\nInitial losses:")
    print(f"Train: {initial['train']:.4f}")
    print(f"Validation: {initial['val']:.4f}")

    # ======================================
    # MAIN TRAINING LOOP
    # ======================================

    for step in range(MAX_ITERS):

        model.train()

        x, y = get_batch(train_data)

        with autocast_context():
            _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became invalid.")

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        # ==================================
        # EVALUATION
        # ==================================

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

                print("Best v2 checkpoint saved!")

    # ======================================
    # COMPLETION
    # ======================================

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

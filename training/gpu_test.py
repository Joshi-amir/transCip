
import torch
import torch.nn as nn
from tokenizers import Tokenizer

import training.train as training
from models.gpt import GPTLanguageModel

print("\nLoading BPE datasets...")

train_data, val_data = training.load_data()

tokenizer = Tokenizer.from_file(
    str(training.TOKENIZER_FILE)
)

vocab_size = tokenizer.get_vocab_size()

print("Vocabulary size:", vocab_size)
print("Training tokens:", len(train_data))

print("\nCreating BPE GPT model...")

model = GPTLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=training.EMBEDDING_DIM,
    num_heads=training.NUM_HEADS,
    num_layers=training.NUM_LAYERS,
    context_length=training.CONTEXT_LENGTH,
    dropout=training.DROPOUT
).cuda()

parameters = sum(
    p.numel() for p in model.parameters()
)

print(f"Parameters: {parameters:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=training.LEARNING_RATE
)

torch.cuda.reset_peak_memory_stats()

print("\nRunning forward pass...")

x, y = training.get_batch(train_data)

with torch.autocast("cuda", dtype=torch.bfloat16):
    _, loss = model(x, y)

print("Loss:", loss.item())

print("\nRunning backward pass...")

optimizer.zero_grad(set_to_none=True)
loss.backward()

nn.utils.clip_grad_norm_(model.parameters(), 1.0)

optimizer.step()

torch.cuda.synchronize()

print("\nGPU test successful!")

print(
    "Peak VRAM:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB"
)

print(
    "Reserved VRAM:",
    round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    "GB"
)

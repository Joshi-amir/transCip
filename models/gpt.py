
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embeddings import InputEmbedding
from models.transformer_block import TransformerBlock


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        context_length=256,
        dropout=0.1
    ):
        super().__init__()

        self.context_length = context_length

        # Convert token IDs into vectors
        self.token_embeddings = InputEmbedding(
            vocab_size,
            context_length,
            embedding_dim
        )

        # Stack multiple Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    context_length,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        # Normalize final representations
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Predict scores for every vocabulary token
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

    def forward(self, token_ids, targets=None):

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError("Input exceeds context length")

        # Embedding
        x = self.token_embeddings(token_ids)

        # Transformer processing
        x = self.transformer_blocks(x)

        # Final normalization
        x = self.final_norm(x)

        # Generate next-token prediction scores
        logits = self.lm_head(x)

        loss = None

        # Calculate training loss if targets are provided
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0):

        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero")

        for _ in range(max_new_tokens):

            # Keep only the latest context window
            context = token_ids[:, -self.context_length:]

            # Get predictions
            logits, _ = self(context)

            # Select scores from the final position
            logits = logits[:, -1, :]

            # Control randomness
            logits = logits / temperature

            # Convert scores into probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append the generated token
            token_ids = torch.cat(
                (token_ids, next_token),
                dim=1
            )

        return token_ids

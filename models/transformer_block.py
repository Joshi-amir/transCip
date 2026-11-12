
import torch
import torch.nn as nn

from models.attention import CausalSelfAttention


class FeedForward(nn.Module):

    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()

        hidden_dim = embedding_dim * 4

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attention = CausalSelfAttention(
            embedding_dim,
            num_heads,
            context_length,
            dropout
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim,
            dropout
        )

    def forward(self, x):

        # Attention with residual connection
        x = x + self.attention(self.norm1(x))

        # Feed-forward network with residual connection
        x = x + self.feed_forward(self.norm2(x))

        return x


import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        dropout=0.1
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0, (
            "Embedding dimension must be divisible by number of heads"
        )

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # One projection creates Query, Key and Value
        self.qkv = nn.Linear(
            embedding_dim,
            3 * embedding_dim
        )

        # Combine the attention heads
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Lower triangular mask prevents looking at future tokens
        mask = torch.tril(
            torch.ones(context_length, context_length)
        )

        self.register_buffer(
            "causal_mask",
            mask.bool(),
            persistent=False
        )

    def forward(self, x):

        batch_size, sequence_length, embedding_dim = x.shape

        if sequence_length > self.causal_mask.size(0):
            raise ValueError("Sequence exceeds context length")

        # Generate Query, Key and Value
        qkv = self.qkv(x)

        # Separate the three representations and attention heads
        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(0)

        # Calculate attention scores
        scores = query @ key.transpose(-2, -1)

        scores = scores * (self.head_dim ** -0.5)

        # Apply causal mask
        mask = self.causal_mask[
            :sequence_length,
            :sequence_length
        ]

        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        # Convert scores into attention probabilities
        weights = torch.softmax(scores, dim=-1)

        weights = self.attention_dropout(weights)

        # Collect information from Value vectors
        output = weights @ value

        # Combine all attention heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embedding_dim
        )

        # Final projection
        output = self.output_projection(output)

        return self.output_dropout(output)

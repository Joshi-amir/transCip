
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)

class PositionalEmbedding(nn.Module):

    def __init__(self, context_length, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim
        )

    def forward(self, positions):
        return self.embedding(positions)

class InputEmbedding(nn.Module):

    def __init__(self, vocab_size, context_length, embedding_dim):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size, embedding_dim
        )

        self.position_embedding = PositionalEmbedding(
            context_length, embedding_dim
        )

    def forward(self, token_ids):

        # Get sequence length
        sequence_length = token_ids.shape[1]

        # Generate position IDs
        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        # Get both embeddings
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        # Combine them
        return token_vectors + position_vectors

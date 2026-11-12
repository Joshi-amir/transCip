
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        dropout=0.1
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(
            embedding_dim,
            3 * embedding_dim
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.attention_dropout = dropout
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch_size, sequence_length, embedding_dim = x.shape

        qkv = self.qkv(x)

        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(0)

        # Optimized causal attention
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True
        )

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embedding_dim
        )

        output = self.output_projection(output)

        return self.output_dropout(output)

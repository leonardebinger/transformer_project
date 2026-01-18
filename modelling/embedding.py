import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Word embedding layer using PyTorch's Embedding."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to embeddings.

        Args:
            x: Token indices of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) as in the paper
        return self.embedding(x) * math.sqrt(self.d_model)

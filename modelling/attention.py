import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, mask_future: bool = False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch, seq_len_q, d_k)
            key: Key tensor of shape (batch, seq_len_k, d_k)
            value: Value tensor of shape (batch, seq_len_k, d_v)
            mask: Padding mask of shape (batch, seq_len_k) where 1 = valid, 0 = pad

        Returns:
            Attention output of shape (batch, seq_len_q, d_v)
        """
        d_k = query.size(-1)

        # Compute attention scores: (batch, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply padding mask if provided
        if mask is not None:
            # Expand mask: (batch, seq_len_k) -> (batch, 1, seq_len_k)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply future/causal mask if needed
        if self.mask_future:
            seq_len_q = query.size(1)
            seq_len_k = key.size(1)
            future_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=query.device), diagonal=1).bool()
            scores = scores.masked_fill(future_mask, float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # Handle NaN from softmax when all values are -inf
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Compute weighted sum of values
        output = torch.matmul(attn_weights, value)

        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, num_heads: int, mask_future: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mask_future = mask_future

        # Linear projections without bias
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask: (batch, seq_len_k) padding mask

        Returns:
            Output of shape (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Project inputs
        q = self.query_transform(query)
        k = self.key_transform(key)
        v = self.value_transform(value)

        # Reshape to (batch, num_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply padding mask
        if mask is not None:
            # (batch, seq_len_k) -> (batch, 1, 1, seq_len_k)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply future/causal mask
        if self.mask_future:
            future_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=query.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(future_mask, float('-inf'))

        # Softmax and handle NaN
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Reshape back: (batch, num_heads, seq_len_q, d_k) -> (batch, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        # Final projection
        output = self.output_transform(context)

        return output

import torch
import torch.nn as nn
import math


class GatedAttention(nn.Module):
    """
    Scaled dot-product attention with output gating.

    The gate is computed from the query and controls how much of the
    attention output flows through.
    """

    def __init__(self, d_k: int, mask_future: bool = False):
        super().__init__()
        self.mask_future = mask_future
        self.d_k = d_k

        # Gate projection: query -> gate values
        self.gate_proj = nn.Linear(d_k, d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None, return_gate: bool = False) -> torch.Tensor:
        """
        Compute gated scaled dot-product attention.

        Args:
            query: (batch, seq_len_q, d_k)
            key: (batch, seq_len_k, d_k)
            value: (batch, seq_len_k, d_v)
            mask: (batch, seq_len_k) padding mask where 1=valid, 0=pad
            return_gate: if True, also return gate values for analysis

        Returns:
            output: (batch, seq_len_q, d_v)
            gate (optional): (batch, seq_len_q, d_v) gate values
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply padding mask
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply future/causal mask
        if self.mask_future:
            seq_len_q, seq_len_k = query.size(1), key.size(1)
            future_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=query.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(future_mask, float('-inf'))

        # Softmax and handle NaN
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Compute attention output
        attention_output = torch.matmul(attn_weights, value)

        # Compute gate from query
        gate = torch.sigmoid(self.gate_proj(query))

        # Apply gate
        gated_output = gate * attention_output

        if return_gate:
            return gated_output, gate
        return gated_output


class GatedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with output gating.

    Same interface as MultiHeadAttention for easy swapping.
    """

    def __init__(self, d_model: int, num_heads: int, mask_future: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mask_future = mask_future

        # Standard projections (no bias, matching existing code)
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        # Gate projection
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None, return_gate: bool = False) -> torch.Tensor:
        """
        Compute gated multi-head attention.

        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask: (batch, seq_len_k) padding mask
            return_gate: if True, also return gate statistics

        Returns:
            output: (batch, seq_len_q, d_model)
            gate_stats (optional): dict with gate statistics for analysis
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

        # Compute attention output
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        # Compute gate from original query
        gate = torch.sigmoid(self.gate_proj(query))

        # Apply gate before output projection
        gated_context = gate * context

        # Final projection
        output = self.output_transform(gated_context)

        if return_gate:
            gate_stats = {
                'gate_mean': gate.mean().item(),
                'gate_std': gate.std().item(),
                'gate_min': gate.min().item(),
                'gate_max': gate.max().item(),
            }
            return output, gate_stats
        return output

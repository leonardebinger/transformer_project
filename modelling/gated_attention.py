"""
Gated Multi-Head Attention with head-specific elementwise gating.

The gate is applied after SDPA output but before reshaping back to d_model,
allowing head-specific learned gating of attention outputs.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict


class GatedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with head-specific elementwise sigmoid gating.

    Gate is applied after SDPA output while still in head-separated form:
    - SDPA output shape: (batch, num_heads, seq_len_q, d_k)
    - Gate shape: (batch, num_heads, seq_len_q, d_k)
    - Gated output = SDPA_output * sigmoid(gate_projection(query))
    """

    def __init__(self, d_model: int, num_heads: int, mask_future: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mask_future = mask_future

        # Linear projections without bias (same as base MultiHeadAttention)
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        # Gate projection with bias (allows learning default gate openness)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        # Initialize gate bias to 0 so sigmoid starts near 0.5
        nn.init.zeros_(self.gate_proj.bias)

    def _compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute head-specific elementwise gate values.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Gate values (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape

        # Project and apply sigmoid
        gate = torch.sigmoid(self.gate_proj(x))  # (batch, seq_len, d_model)

        # Reshape to head-specific form
        gate = gate.view(batch_size, seq_len, self.num_heads, self.d_k)
        gate = gate.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

        return gate

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_gate: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute gated multi-head attention.

        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_k, d_model)
            mask: (batch, seq_len_k) padding mask
            return_gate: If True, return gate statistics

        Returns:
            Output of shape (batch, seq_len_q, d_model)
            Optionally: gate statistics dict
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

        # Apply attention to values: (batch, num_heads, seq_len_q, d_k)
        context = torch.matmul(attn_weights, v)

        # === GATING: Apply head-specific elementwise gate ===
        gate = self._compute_gate(query)  # (batch, num_heads, seq_len_q, d_k)
        context = context * gate

        # Collect gate statistics if requested
        gate_stats = None
        if return_gate:
            gate_stats = {
                'gate_mean': gate.mean().item(),
                'gate_std': gate.std().item(),
                'gate_min': gate.min().item(),
                'gate_max': gate.max().item(),
            }

        # Reshape back: (batch, num_heads, seq_len_q, d_k) -> (batch, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        # Final projection
        output = self.output_transform(context)

        if return_gate:
            return output, gate_stats
        return output

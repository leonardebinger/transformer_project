"""
Gated Transformer: Identical to base Transformer but uses GatedMultiHeadAttention.

This module provides a direct comparison point for standard vs gated attention.
All architecture choices mirror modelling/transformer.py exactly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any, List

from modelling.positional_encoding import PositionalEncoding
from modelling.gated_attention import GatedMultiHeadAttention


class GatedTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with gated multi-head attention.
    Mirrors BaseTransformerLayer from modelling/functional.py.
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        # Gated self-attention (replaces MultiHeadAttention)
        self.self_attention = GatedMultiHeadAttention(d_model, num_heads, mask_future=False)

        # Position-wise feed-forward (same structure as base)
        self.feature_transformation = nn.Sequential()
        self.feature_transformation.add_module('linear1', nn.Linear(d_model, dim_feedforward))
        self.feature_transformation.add_module('relu', nn.ReLU())
        self.feature_transformation.add_module('linear2', nn.Linear(dim_feedforward, d_model))

        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_gate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Padding mask (batch, seq_len)
            return_gate: If True, return gate statistics

        Returns:
            Output tensor (batch, seq_len, d_model)
            Optionally: gate statistics dict
        """
        # Self-attention with residual and layer norm
        if return_gate:
            attn_output, gate_stats = self.self_attention(x, x, x, mask, return_gate=True)
        else:
            attn_output = self.self_attention(x, x, x, mask)
            gate_stats = None

        x = self.layer_norm_1(x + self.dropout(attn_output))

        # Feed-forward with residual and layer norm
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_2(x + self.dropout(ff_output))

        if return_gate:
            return x, gate_stats
        return x


class GatedTransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with gated multi-head attention.
    Mirrors TransformerDecoderLayer from modelling/functional.py.
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        # Gated masked self-attention
        self.self_attention = GatedMultiHeadAttention(d_model, num_heads, mask_future=True)

        # Gated encoder cross-attention
        self.encoder_attention = GatedMultiHeadAttention(d_model, num_heads, mask_future=False)

        # Position-wise feed-forward
        self.feature_transformation = nn.Sequential()
        self.feature_transformation.add_module('linear1', nn.Linear(d_model, dim_feedforward))
        self.feature_transformation.add_module('relu', nn.ReLU())
        self.feature_transformation.add_module('linear2', nn.Linear(dim_feedforward, d_model))

        # Three layer normalizations
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_gate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Args:
            x: Decoder input (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            memory_mask: Padding mask for encoder output (batch, src_seq_len)
            tgt_mask: Padding mask for decoder input (batch, tgt_seq_len)
            return_gate: If True, return gate statistics

        Returns:
            Output tensor (batch, tgt_seq_len, d_model)
            Optionally: gate statistics dict
        """
        gate_stats = {}

        # Masked self-attention with residual and layer norm
        if return_gate:
            self_attn_output, gate_stats['self'] = self.self_attention(
                x, x, x, tgt_mask, return_gate=True
            )
        else:
            self_attn_output = self.self_attention(x, x, x, tgt_mask)

        x = self.layer_norm_1(x + self.dropout(self_attn_output))

        # Cross-attention with residual and layer norm
        if return_gate:
            cross_attn_output, gate_stats['cross'] = self.encoder_attention(
                x, encoder_output, encoder_output, memory_mask, return_gate=True
            )
        else:
            cross_attn_output = self.encoder_attention(x, encoder_output, encoder_output, memory_mask)

        x = self.layer_norm_2(x + self.dropout(cross_attn_output))

        # Feed-forward with residual and layer norm
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_3(x + self.dropout(ff_output))

        if return_gate:
            return x, gate_stats
        return x


class GatedTransformer(nn.Module):
    """
    Transformer with gated attention mechanism.
    Structurally identical to Transformer from modelling/transformer.py.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Shared embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Gated encoder layers
        self.encoder_layers = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Gated decoder layers
        self.decoder_layers = nn.ModuleList([
            GatedTransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection (shares weights with embedding)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_gate: bool = False
    ) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: Source token ids (batch, src_len)
            src_mask: Source padding mask (batch, src_len)
            return_gate: If True, return gate statistics

        Returns:
            Encoder output (batch, src_len, d_model)
            Optionally: list of gate statistics per layer
        """
        # Embed and add positional encoding (same scaling as base)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        gate_stats_list = []
        for layer in self.encoder_layers:
            if return_gate:
                x, gate_stats = layer(x, src_mask, return_gate=True)
                gate_stats_list.append(gate_stats)
            else:
                x = layer(x, src_mask)

        if return_gate:
            return x, gate_stats_list
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_gate: bool = False
    ) -> torch.Tensor:
        """
        Decode target sequence.

        Args:
            tgt: Target token ids (batch, tgt_len)
            encoder_output: Encoder output (batch, src_len, d_model)
            src_mask: Source padding mask (batch, src_len)
            tgt_mask: Target padding mask (batch, tgt_len)
            return_gate: If True, return gate statistics

        Returns:
            Decoder output (batch, tgt_len, d_model)
            Optionally: list of gate statistics per layer
        """
        # Embed and add positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        gate_stats_list = []
        for layer in self.decoder_layers:
            if return_gate:
                x, gate_stats = layer(x, encoder_output, src_mask, tgt_mask, return_gate=True)
                gate_stats_list.append(gate_stats)
            else:
                x = layer(x, encoder_output, src_mask, tgt_mask)

        if return_gate:
            return x, gate_stats_list
        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            src: Source token ids (batch, src_len)
            tgt: Target token ids (batch, tgt_len)
            src_mask: Source padding mask (batch, src_len)
            tgt_mask: Target padding mask (batch, tgt_len)

        Returns:
            Output logits (batch, tgt_len, vocab_size)
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits

    def forward_with_gate_stats(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass that also returns aggregated gate statistics.

        Returns:
            (logits, gate_stats_dict)
        """
        encoder_output, enc_gate_stats = self.encode(src, src_mask, return_gate=True)
        decoder_output, dec_gate_stats = self.decode(
            tgt, encoder_output, src_mask, tgt_mask, return_gate=True
        )
        logits = self.output_projection(decoder_output)

        # Aggregate gate statistics
        all_gate_means = []
        all_gate_stds = []

        for stats in enc_gate_stats:
            if stats:
                all_gate_means.append(stats['gate_mean'])
                all_gate_stds.append(stats['gate_std'])

        for stats in dec_gate_stats:
            if 'self' in stats and stats['self']:
                all_gate_means.append(stats['self']['gate_mean'])
                all_gate_stds.append(stats['self']['gate_std'])
            if 'cross' in stats and stats['cross']:
                all_gate_means.append(stats['cross']['gate_mean'])
                all_gate_stds.append(stats['cross']['gate_std'])

        aggregated_stats = {
            'gate_mean': sum(all_gate_means) / len(all_gate_means) if all_gate_means else 0.5,
            'gate_std': sum(all_gate_stds) / len(all_gate_stds) if all_gate_stds else 0.0,
        }

        return logits, aggregated_stats

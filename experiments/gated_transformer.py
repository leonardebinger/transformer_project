import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import math

from modelling.positional_encoding import PositionalEncoding
from modelling.gated_attention import GatedMultiHeadAttention


class GatedTransformerEncoderLayer(nn.Module):
    """Encoder layer with gated multi-head attention."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = GatedMultiHeadAttention(d_model, num_heads, mask_future=False)

        self.feature_transformation = nn.Sequential()
        self.feature_transformation.add_module('linear1', nn.Linear(d_model, dim_feedforward))
        self.feature_transformation.add_module('relu', nn.ReLU())
        self.feature_transformation.add_module('linear2', nn.Linear(dim_feedforward, d_model))

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_gate=False):
        if return_gate:
            attn_output, gate_stats = self.self_attention(x, x, x, mask, return_gate=True)
        else:
            attn_output = self.self_attention(x, x, x, mask)
            gate_stats = None

        x = self.layer_norm_1(x + self.dropout(attn_output))
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_2(x + self.dropout(ff_output))

        if return_gate:
            return x, gate_stats
        return x


class GatedTransformerDecoderLayer(nn.Module):
    """Decoder layer with gated multi-head attention."""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = GatedMultiHeadAttention(d_model, num_heads, mask_future=True)
        self.encoder_attention = GatedMultiHeadAttention(d_model, num_heads, mask_future=False)

        self.feature_transformation = nn.Sequential()
        self.feature_transformation.add_module('linear1', nn.Linear(d_model, dim_feedforward))
        self.feature_transformation.add_module('relu', nn.ReLU())
        self.feature_transformation.add_module('linear2', nn.Linear(dim_feedforward, d_model))

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, memory_mask=None, tgt_mask=None, return_gate=False):
        gate_stats = {}

        if return_gate:
            self_attn_output, gate_stats['self'] = self.self_attention(x, x, x, tgt_mask, return_gate=True)
        else:
            self_attn_output = self.self_attention(x, x, x, tgt_mask)

        x = self.layer_norm_1(x + self.dropout(self_attn_output))

        if return_gate:
            cross_attn_output, gate_stats['cross'] = self.encoder_attention(
                x, encoder_output, encoder_output, memory_mask, return_gate=True
            )
        else:
            cross_attn_output = self.encoder_attention(x, encoder_output, encoder_output, memory_mask)

        x = self.layer_norm_2(x + self.dropout(cross_attn_output))
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_3(x + self.dropout(ff_output))

        if return_gate:
            return x, gate_stats
        return x


class GatedTransformer(nn.Module):
    """Transformer with gated attention mechanism."""

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

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            GatedTransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying with embedding
        self.output_projection.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None, return_gate=False):
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

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None, return_gate=False):
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

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits

    def forward_with_gate_stats(self, src, tgt, src_mask=None, tgt_mask=None):
        """Forward pass that also returns gate statistics."""
        encoder_output, enc_gate_stats = self.encode(src, src_mask, return_gate=True)
        decoder_output, dec_gate_stats = self.decode(tgt, encoder_output, src_mask, tgt_mask, return_gate=True)
        logits = self.output_projection(decoder_output)

        # Collect per-layer stats and aggregate
        per_layer = {'encoder': [], 'decoder': []}
        all_gate_means = []
        all_gate_stds = []

        for i, stats in enumerate(enc_gate_stats):
            if stats:
                per_layer['encoder'].append({
                    'layer': i,
                    'self': {
                        'gate_mean': stats['gate_mean'],
                        'gate_std': stats['gate_std'],
                        'gate_min': stats['gate_min'],
                        'gate_max': stats['gate_max'],
                    }
                })
                all_gate_means.append(stats['gate_mean'])
                all_gate_stds.append(stats['gate_std'])

        for i, stats in enumerate(dec_gate_stats):
            layer_entry = {'layer': i}
            if 'self' in stats:
                layer_entry['self'] = {
                    'gate_mean': stats['self']['gate_mean'],
                    'gate_std': stats['self']['gate_std'],
                    'gate_min': stats['self']['gate_min'],
                    'gate_max': stats['self']['gate_max'],
                }
                all_gate_means.append(stats['self']['gate_mean'])
                all_gate_stds.append(stats['self']['gate_std'])
            if 'cross' in stats:
                layer_entry['cross'] = {
                    'gate_mean': stats['cross']['gate_mean'],
                    'gate_std': stats['cross']['gate_std'],
                    'gate_min': stats['cross']['gate_min'],
                    'gate_max': stats['cross']['gate_max'],
                }
                all_gate_means.append(stats['cross']['gate_mean'])
                all_gate_stds.append(stats['cross']['gate_std'])
            per_layer['decoder'].append(layer_entry)

        aggregated_stats = {
            'gate_mean': sum(all_gate_means) / len(all_gate_means) if all_gate_means else 0,
            'gate_std': sum(all_gate_stds) / len(all_gate_stds) if all_gate_stds else 0,
            'per_layer': per_layer,
        }

        return logits, aggregated_stats

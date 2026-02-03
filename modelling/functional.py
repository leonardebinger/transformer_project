import torch
import torch.nn as nn

from modelling.attention import MultiHeadAttention


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network: FFN(x) = ReLU(xW1 + b1)W2 + b2"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class BaseTransformerLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward network."""

    def __init__(self, input_dim: int, num_heads: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False)

        # Position-wise feed-forward using Sequential with named modules
        self.feature_transformation = nn.Sequential()
        self.feature_transformation.add_module('linear1', nn.Linear(input_dim, feature_dim))
        self.feature_transformation.add_module('relu', nn.ReLU())
        self.feature_transformation.add_module('linear2', nn.Linear(feature_dim, input_dim))

        # Layer normalization (two independent instances)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Padding mask of shape (batch, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.layer_norm_1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_2(x + self.dropout(ff_output))

        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with masked self-attention, cross-attention, and FFN."""

    def __init__(self, input_dim: int, num_heads: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()

        # Masked self-attention (with future masking)
        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True)

        # Encoder cross-attention
        self.encoder_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False)

        # Position-wise feed-forward
        self.feature_transformation = nn.Sequential()
        self.feature_transformation.add_module('linear1', nn.Linear(input_dim, feature_dim))
        self.feature_transformation.add_module('relu', nn.ReLU())
        self.feature_transformation.add_module('linear2', nn.Linear(feature_dim, input_dim))

        # Three layer normalizations
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                memory_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Args:
            x: Decoder input of shape (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch, src_seq_len, d_model)
            memory_mask: Padding mask for encoder output (batch, src_seq_len)
            tgt_mask: Padding mask for decoder input (batch, tgt_seq_len)

        Returns:
            Output tensor of shape (batch, tgt_seq_len, d_model)
        """
        # Masked self-attention with residual and layer norm
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.layer_norm_1(x + self.dropout(self_attn_output))

        # Encoder cross-attention with residual and layer norm
        cross_attn_output = self.encoder_attention(x, encoder_output, encoder_output, memory_mask)
        x = self.layer_norm_2(x + self.dropout(cross_attn_output))

        # Feed-forward with residual and layer norm
        ff_output = self.feature_transformation(x)
        x = self.layer_norm_3(x + self.dropout(ff_output))

        return x

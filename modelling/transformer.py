import torch
import torch.nn as nn
import math

from modelling.attention import MultiHeadAttention
from modelling.positional_encoding import PositionalEncoding
from modelling.functional import BaseTransformerLayer, TransformerDecoderLayer


class Transformer(nn.Module):
    """Full Transformer model for sequence-to-sequence tasks."""

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

        # Shared embedding layer (used for both encoder and decoder input)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            BaseTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection (shares weights with embedding as per paper)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: Source token ids (batch, src_len)
            src_mask: Source padding mask (batch, src_len)

        Returns:
            Encoder output (batch, src_len, d_model)
        """
        # Embed and add positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Decode target sequence.

        Args:
            tgt: Target token ids (batch, tgt_len)
            encoder_output: Encoder output (batch, src_len, d_model)
            src_mask: Source padding mask (batch, src_len)
            tgt_mask: Target padding mask (batch, tgt_len)

        Returns:
            Decoder output (batch, tgt_len, d_model)
        """
        # Embed and add positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
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
        # Encode source
        encoder_output = self.encode(src, src_mask)

        # Decode target
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

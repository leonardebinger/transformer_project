"""
Local testing script using synthetic data (no GPU required).
For actual WMT17 DE-EN training, see train_wmt.py instead.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import random

from modelling.transformer import Transformer
from modelling.scheduler import TransformerLRScheduler, get_optimizer
from modelling.generation import greedy_decode, translate_sentence, translate_batch, compute_bleu
from dataset import (
    TranslationDataset, Vocabulary, collate_fn,
    clean_text, is_valid_pair, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
)


def create_synthetic_data(num_samples=1000, vocab_size=500, max_len=20):
    """Create synthetic translation data for testing."""
    data = []
    for _ in range(num_samples):
        src_len = random.randint(5, max_len)
        tgt_len = random.randint(5, max_len)
        # Create random "sentences" as space-separated tokens
        src = ' '.join([f'word{random.randint(0, vocab_size-1)}' for _ in range(src_len)])
        tgt = ' '.join([f'word{random.randint(0, vocab_size-1)}' for _ in range(tgt_len)])
        data.append((src, tgt))
    return data


def build_vocab_from_data(data, vocab_size=500):
    """Build vocabulary from data pairs."""
    vocab = Vocabulary(vocab_size=vocab_size)
    all_texts = [src for src, _ in data] + [tgt for _, tgt in data]
    vocab.build(all_texts)
    return vocab


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)

        # Decoder input: all tokens except last
        # Decoder target: all tokens except first (shifted by 1)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask_input = tgt_mask[:, :-1]

        # Forward pass
        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask_input)

        # Reshape for loss calculation
        # logits: (batch, tgt_len-1, vocab_size) -> (batch * (tgt_len-1), vocab_size)
        # tgt_output: (batch, tgt_len-1) -> (batch * (tgt_len-1),)
        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)

        # Calculate loss (ignore padding)
        loss = criterion(logits, tgt_output)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask_input = tgt_mask[:, :-1]

            logits = model(src, tgt_input, src_mask, tgt_mask_input)

            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(logits, tgt_output)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    # Hyperparameters
    vocab_size = 500
    d_model = 64
    n_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 128
    dropout = 0.1
    max_len = 64

    batch_size = 32
    num_epochs = 5
    warmup_steps = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create synthetic data
    print("Creating synthetic data...")
    train_data = create_synthetic_data(num_samples=1000, vocab_size=vocab_size)
    val_data = create_synthetic_data(num_samples=200, vocab_size=vocab_size)

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab_from_data(train_data + val_data, vocab_size=vocab_size)
    print(f"Vocabulary size: {len(vocab)}")

    # Create datasets
    train_dataset = TranslationDataset(train_data, vocab, max_len=max_len)
    val_dataset = TranslationDataset(val_data, vocab, max_len=max_len)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Initialize model
    print("Initializing model...")
    model = Transformer(
        vocab_size=len(vocab),
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model, lr=1e-4, weight_decay=0.01)
    scheduler = TransformerLRScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    # Loss function (ignore padding token)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, criterion, device)
        val_loss = evaluate(model, val_dataloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_lr():.6f}")

    # Verify loss decreased
    print("\nTraining Summary:")
    print(f"Initial train loss: {train_losses[0]:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Initial val loss: {val_losses[0]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")

    if train_losses[-1] < train_losses[0]:
        print("✓ Training loss decreased")
    else:
        print("✗ Training loss did not decrease")

    if val_losses[-1] < val_losses[0]:
        print("✓ Validation loss decreased")
    else:
        print("✗ Validation loss did not decrease")

    # Evaluation with generation and BLEU
    print("\n" + "="*50)
    print("Evaluation: Generation and BLEU Score")
    print("="*50)

    # Generate translations for validation set
    print("\nGenerating translations...")
    test_sources = [src for src, _ in val_data[:20]]
    test_references = [tgt for _, tgt in val_data[:20]]

    predictions = translate_batch(model, test_sources, vocab, max_len=max_len, device=device)

    # Show some examples
    print("\nSample translations:")
    for i in range(min(5, len(predictions))):
        print(f"\n  Source:     {test_sources[i][:60]}...")
        print(f"  Reference:  {test_references[i][:60]}...")
        print(f"  Prediction: {predictions[i][:60]}...")

    # Compute BLEU score
    print("\nComputing BLEU score...")
    bleu_results = compute_bleu(predictions, test_references)
    print(f"BLEU Score: {bleu_results['bleu']:.4f}")
    print(f"Precisions: {[f'{p:.4f}' for p in bleu_results['precisions']]}")

    return model, train_losses, val_losses, vocab


if __name__ == '__main__':
    main()

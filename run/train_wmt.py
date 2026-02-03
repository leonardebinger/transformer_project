"""
WMT17 German-English Translation Training Script

This script trains a transformer model on the WMT17 German-English dataset
as specified in the practicals, with result saving and evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from datetime import datetime
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import tqdm

from modelling.transformer import Transformer
from modelling.scheduler import TransformerLRScheduler, get_optimizer
from modelling.tokenizer import HuggingFaceBPETokenizer
from modelling.generation import greedy_decode, compute_bleu
from dataset import clean_text, is_valid_pair, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


# Configuration
CONFIG = {
    # Model - "Base" configuration from "Attention Is All You Need"
    'vocab_size': 50000,
    'd_model': 512,
    'n_heads': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'max_len': 128,

    # Training - following "Attention Is All You Need" paper, optimized for A100 40GB
    'batch_size': 128,
    'gradient_accumulation': 4,    # Effective batch = 128*4 = 512 sentences (~15k tokens)
    'num_epochs': 5,
    'warmup_steps': 4000,
    'max_grad_norm': 1.0,
    'label_smoothing': 0.1,        # Paper uses 0.1 - crucial for translation quality

    # Data - use full WMT17 dataset
    'train_subset_size': None,  # None = use full dataset
    'val_subset_size': 10000,
    'test_subset_size': 3000,
    'min_length': 5,
    'max_length': 128,
}


class WMTDataset(torch.utils.data.Dataset):
    """Dataset for WMT translation pairs using BPE tokenizer."""

    def __init__(self, data: List[Tuple[str, str]], tokenizer: HuggingFaceBPETokenizer,
                 max_len: int = 64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]

        # Encode with BPE tokenizer (adds BOS/EOS for both source and target)
        src_ids = self.tokenizer.encode(src, add_special_tokens=True)
        tgt_ids = self.tokenizer.encode(tgt, add_special_tokens=True)

        # Truncate
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]

        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'src_len': len(src_ids),
            'tgt_len': len(tgt_ids),
            'src_text': src,
            'tgt_text': tgt
        }


def collate_fn(batch, pad_idx=0):
    """Collate function with padding."""
    max_src_len = max(item['src_len'] for item in batch)
    max_tgt_len = max(item['tgt_len'] for item in batch)

    src_padded = []
    tgt_padded = []
    src_mask = []
    tgt_mask = []

    for item in batch:
        src = item['src']
        src_pad_len = max_src_len - len(src)
        src_padded.append(torch.cat([src, torch.full((src_pad_len,), pad_idx, dtype=torch.long)]))
        src_mask.append(torch.cat([torch.ones(len(src)), torch.zeros(src_pad_len)]))

        tgt = item['tgt']
        tgt_pad_len = max_tgt_len - len(tgt)
        tgt_padded.append(torch.cat([tgt, torch.full((tgt_pad_len,), pad_idx, dtype=torch.long)]))
        tgt_mask.append(torch.cat([torch.ones(len(tgt)), torch.zeros(tgt_pad_len)]))

    return {
        'src': torch.stack(src_padded),
        'tgt': torch.stack(tgt_padded),
        'src_mask': torch.stack(src_mask),
        'tgt_mask': torch.stack(tgt_mask)
    }


def load_and_clean_wmt17(split: str, max_samples: int = None) -> List[Tuple[str, str]]:
    """Load and clean WMT17 dataset."""
    print(f"Loading WMT17 {split} split...")
    dataset = load_dataset("wmt17", "de-en", split=split, trust_remote_code=True)

    cleaned = []
    for item in tqdm.tqdm(dataset, desc=f"Cleaning {split}"):
        src = clean_text(item['translation']['de'])
        tgt = clean_text(item['translation']['en'])

        if is_valid_pair(src, tgt, min_len=CONFIG['min_length'], max_len=CONFIG['max_length']):
            cleaned.append((src, tgt))

        if max_samples and len(cleaned) >= max_samples:
            break

    print(f"  Cleaned {len(cleaned)} pairs from {split}")
    return cleaned


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch, num_epochs, scaler=None):
    """Train for one epoch with gradient accumulation and optional mixed precision."""
    model.train()
    total_loss = 0
    num_batches = 0
    use_amp = scaler is not None
    accum_steps = CONFIG['gradient_accumulation']

    optimizer.zero_grad()
    progress = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, batch in enumerate(progress):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask_input = tgt_mask[:, :-1]

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(src, tgt_input, src_mask, tgt_mask_input)
                logits = logits.reshape(-1, logits.size(-1))
                tgt_output_flat = tgt_output.reshape(-1)
                loss = criterion(logits, tgt_output_flat) / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            logits = model(src, tgt_input, src_mask, tgt_mask_input)
            logits = logits.reshape(-1, logits.size(-1))
            tgt_output_flat = tgt_output.reshape(-1)
            loss = criterion(logits, tgt_output_flat) / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        total_loss += loss.item() * accum_steps  # Unscale for logging
        num_batches += 1
        progress.set_postfix({'loss': loss.item() * accum_steps, 'lr': scheduler.get_lr()})

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
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


SAMPLE_SENTENCES = [
    "Ich liebe dich.",
    "Wo ist der Bahnhof?",
    "Das Wetter ist heute schön.",
    "Ich möchte ein Bier bestellen.",
    "Wie viel kostet das?",
]


def translate_sentence(model, sentence, tokenizer, device, max_len=128):
    """Translate a single sentence."""
    model.eval()
    with torch.no_grad():
        src_ids = tokenizer.encode(sentence, add_special_tokens=True)
        src_ids = src_ids[:max_len]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = torch.ones(1, len(src_ids), device=device)

        output_ids = greedy_decode(
            model, src, src_mask,
            bos_idx=tokenizer.bos_idx,
            eos_idx=tokenizer.eos_idx,
            max_len=max_len,
            device=device
        )

        output_ids = output_ids[0].tolist()
        if tokenizer.bos_idx in output_ids:
            output_ids = output_ids[output_ids.index(tokenizer.bos_idx) + 1:]
        if tokenizer.eos_idx in output_ids:
            output_ids = output_ids[:output_ids.index(tokenizer.eos_idx)]

        return tokenizer.decode(output_ids, skip_special_tokens=True)


def print_sample_translations(model, test_data, tokenizer, device, epoch):
    """Print sample translations after each epoch."""
    print(f"\n  Sample Translations (Epoch {epoch+1}):")
    print("  " + "-"*50)

    # 10 sentences from test set
    print("  From test set:")
    for i in range(min(10, len(test_data))):
        src, ref = test_data[i]
        pred = translate_sentence(model, src, tokenizer, device)
        print(f"    DE: {src[:60]}...")
        print(f"    EN: {pred[:60]}...")
        print()

    # 5 hardcoded sentences
    print("  Common phrases:")
    for src in SAMPLE_SENTENCES:
        pred = translate_sentence(model, src, tokenizer, device)
        print(f"    DE: {src}")
        print(f"    EN: {pred}")
        print()


def generate_translations(model, test_data, tokenizer, device, max_samples=100):
    """Generate translations for test data."""
    model.eval()
    predictions = []
    references = []

    for src_text, tgt_text in tqdm.tqdm(test_data[:max_samples], desc="Generating"):
        # Encode source with special tokens and truncate to max_len
        src_ids = tokenizer.encode(src_text, add_special_tokens=True)
        src_ids = src_ids[:CONFIG['max_len']]  # Truncate to avoid exceeding positional encoding
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = torch.ones(1, len(src_ids), device=device)

        # Generate
        output_ids = greedy_decode(
            model, src, src_mask,
            bos_idx=tokenizer.bos_idx,
            eos_idx=tokenizer.eos_idx,
            max_len=CONFIG['max_len'],
            device=device
        )

        # Decode output
        output_ids = output_ids[0].tolist()
        if tokenizer.bos_idx in output_ids:
            output_ids = output_ids[output_ids.index(tokenizer.bos_idx) + 1:]
        if tokenizer.eos_idx in output_ids:
            output_ids = output_ids[:output_ids.index(tokenizer.eos_idx)]

        prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
        predictions.append(prediction)
        references.append(tgt_text)

    return predictions, references


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("\n" + "="*60)
    print("Loading WMT17 Dataset")
    print("="*60)

    train_data = load_and_clean_wmt17('train', CONFIG['train_subset_size'])
    val_data = load_and_clean_wmt17('validation', CONFIG['val_subset_size'])
    test_data = load_and_clean_wmt17('test', CONFIG['test_subset_size'])

    # Train tokenizer
    print("\n" + "="*60)
    print("Training BPE Tokenizer")
    print("="*60)

    all_texts = [src for src, _ in train_data] + [tgt for _, tgt in train_data]
    tokenizer = HuggingFaceBPETokenizer(vocab_size=CONFIG['vocab_size'])
    tokenizer.train(all_texts)
    print(f"Vocabulary size: {tokenizer.vocab_size_actual}")

    # Save tokenizer
    tokenizer_dir = os.path.join(results_dir, f'tokenizer_{timestamp}')
    tokenizer.save(tokenizer_dir)
    print(f"Tokenizer saved to: {tokenizer_dir}")

    # Create datasets
    train_dataset = WMTDataset(train_data, tokenizer, CONFIG['max_len'])
    val_dataset = WMTDataset(val_data, tokenizer, CONFIG['max_len'])

    pad_idx = tokenizer.pad_idx
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx),
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx),
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    print("\n" + "="*60)
    print("Initializing Model")
    print("="*60)

    model = Transformer(
        vocab_size=tokenizer.vocab_size_actual,
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        num_encoder_layers=CONFIG['num_encoder_layers'],
        num_decoder_layers=CONFIG['num_decoder_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        dropout=CONFIG['dropout'],
        max_len=CONFIG['max_len']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model, lr=1e-4, weight_decay=0.01)
    scheduler = TransformerLRScheduler(optimizer, d_model=CONFIG['d_model'],
                                        warmup_steps=CONFIG['warmup_steps'])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=CONFIG['label_smoothing'])

    # Training
    print("\n" + "="*60)
    print("Training")
    print("="*60)

    # Initialize mixed precision scaler for CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    if scaler:
        print("Using mixed precision training (AMP)")

    results = {
        'config': CONFIG,
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'timestamp': timestamp
    }

    for epoch in range(CONFIG['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                  criterion, device, epoch, CONFIG['num_epochs'], scaler)
        val_loss = evaluate(model, val_loader, criterion, device)

        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['learning_rates'].append(scheduler.get_lr())

        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {scheduler.get_lr():.6f}")

        # Sample translations after each epoch
        print_sample_translations(model, test_data, tokenizer, device, epoch)

    # Training summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Initial train loss: {results['train_losses'][0]:.4f}")
    print(f"Final train loss: {results['train_losses'][-1]:.4f}")
    print(f"Initial val loss: {results['val_losses'][0]:.4f}")
    print(f"Final val loss: {results['val_losses'][-1]:.4f}")

    if results['train_losses'][-1] < results['train_losses'][0]:
        print("Training loss decreased")
    if results['val_losses'][-1] < results['val_losses'][0]:
        print("Validation loss decreased")

    # Save model
    model_path = os.path.join(results_dir, f'model_{timestamp}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    # Evaluation on test set
    print("\n" + "="*60)
    print("Evaluation on WMT17 Test Set")
    print("="*60)

    predictions, references = generate_translations(
        model, test_data, tokenizer, device, max_samples=CONFIG['test_subset_size']
    )

    # BLEU score
    bleu_results = compute_bleu(predictions, references)
    results['bleu'] = bleu_results['bleu']
    results['bleu_precisions'] = bleu_results['precisions']

    print(f"\nBLEU Score: {bleu_results['bleu']:.4f}")
    print(f"Precisions: {[f'{p:.4f}' for p in bleu_results['precisions']]}")

    # Sample translations
    print("\nSample Translations:")
    results['sample_translations'] = []
    for i in range(min(10, len(predictions))):
        sample = {
            'source': test_data[i][0],
            'reference': references[i],
            'prediction': predictions[i]
        }
        results['sample_translations'].append(sample)
        print(f"\n  Source (DE):    {test_data[i][0][:80]}...")
        print(f"  Reference (EN): {references[i][:80]}...")
        print(f"  Prediction:     {predictions[i][:80]}...")

    # Save results
    results_file = os.path.join(results_dir, f'results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == '__main__':
    main()

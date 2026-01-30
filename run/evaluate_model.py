"""
Evaluate a trained model on WMT17 test set.

Usage:
    python run/evaluate_model.py --model results/model_TIMESTAMP.pt --tokenizer results/tokenizer_TIMESTAMP/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime

import torch
import tqdm

from modelling.transformer import Transformer
from modelling.tokenizer import HuggingFaceBPETokenizer
from modelling.generation import greedy_decode, compute_bleu
from dataset import clean_text, is_valid_pair
from datasets import load_dataset


CONFIG = {
    'd_model': 64,
    'n_heads': 4,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'dim_feedforward': 128,
    'dropout': 0.1,
    'max_len': 64,
    'min_length': 5,
    'max_length': 64,
    'test_subset_size': 1000,
}


def load_test_data(max_samples):
    """Load and clean WMT17 test data."""
    print("Loading WMT17 test split...")
    dataset = load_dataset("wmt17", "de-en", split="test")

    cleaned = []
    for item in tqdm.tqdm(dataset, desc="Cleaning test"):
        src = clean_text(item['translation']['de'])
        tgt = clean_text(item['translation']['en'])

        if is_valid_pair(src, tgt, min_len=CONFIG['min_length'], max_len=CONFIG['max_length']):
            cleaned.append((src, tgt))

        if max_samples and len(cleaned) >= max_samples:
            break

    print(f"  Cleaned {len(cleaned)} pairs")
    return cleaned


def generate_translations(model, test_data, tokenizer, device, max_len, max_samples=100):
    """Generate translations for test data."""
    model.eval()
    predictions = []
    references = []

    for src_text, tgt_text in tqdm.tqdm(test_data[:max_samples], desc="Generating"):
        # Encode source with special tokens and truncate
        src_ids = tokenizer.encode(src_text, add_special_tokens=True)
        src_ids = src_ids[:max_len]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = torch.ones(1, len(src_ids), device=device)

        # Generate
        output_ids = greedy_decode(
            model, src, src_mask,
            bos_idx=tokenizer.bos_idx,
            eos_idx=tokenizer.eos_idx,
            max_len=max_len,
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
    parser = argparse.ArgumentParser(description='Evaluate trained model on WMT17')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer directory')
    parser.add_argument('--samples', type=int, default=CONFIG['test_subset_size'],
                        help='Number of test samples')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer from: {args.tokenizer}")
    tokenizer = HuggingFaceBPETokenizer.load(args.tokenizer)
    print(f"Vocabulary size: {tokenizer.vocab_size_actual}")

    # Load model
    print(f"\nLoading model from: {args.model}")
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

    model.load_state_dict(torch.load(args.model, map_location=device))
    print("Model loaded successfully")

    # Load test data
    test_data = load_test_data(args.samples)

    # Generate translations
    print("\n" + "="*60)
    print("Generating Translations")
    print("="*60)

    predictions, references = generate_translations(
        model, test_data, tokenizer, device, CONFIG['max_len'], args.samples
    )

    # Compute BLEU
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    bleu_results = compute_bleu(predictions, references)
    print(f"\nBLEU Score: {bleu_results['bleu']:.4f}")
    print(f"Precisions: {[f'{p:.4f}' for p in bleu_results['precisions']]}")

    # Sample translations
    print("\nSample Translations:")
    for i in range(min(10, len(predictions))):
        print(f"\n  Source (DE):    {test_data[i][0][:80]}...")
        print(f"  Reference (EN): {references[i][:80]}...")
        print(f"  Prediction:     {predictions[i][:80]}...")

    # Save results
    results_dir = os.path.dirname(args.model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'model_path': args.model,
        'tokenizer_path': args.tokenizer,
        'bleu': bleu_results['bleu'],
        'bleu_precisions': bleu_results['precisions'],
        'num_samples': len(predictions),
        'sample_translations': [
            {'source': test_data[i][0], 'reference': references[i], 'prediction': predictions[i]}
            for i in range(min(10, len(predictions)))
        ]
    }

    results_file = os.path.join(results_dir, f'eval_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()

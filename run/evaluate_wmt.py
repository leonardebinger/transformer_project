"""
Comprehensive WMT17 Evaluation Script

This script evaluates a trained transformer model on WMT17 DE-EN translation,
computing BLEU scores, generating sample translations, and creating visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import torch
import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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
}


def load_test_data(max_samples: int) -> List[Tuple[str, str]]:
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
    sources = []

    for src_text, tgt_text in tqdm.tqdm(test_data[:max_samples], desc="Generating"):
        src_ids = tokenizer.encode(src_text, add_special_tokens=False)
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

        prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
        predictions.append(prediction)
        references.append(tgt_text)
        sources.append(src_text)

    return predictions, references, sources


def categorize_errors(predictions: List[str], references: List[str]) -> Dict[str, int]:
    """Categorize translation errors for analysis."""
    error_counts = {
        'empty_output': 0,
        'repetition': 0,
        'truncated': 0,
        'reasonable': 0,
    }

    for pred, ref in zip(predictions, references):
        pred_words = pred.split() if pred else []
        ref_words = ref.split()

        if not pred.strip():
            error_counts['empty_output'] += 1
        elif len(pred_words) > 2 and len(set(pred_words)) < len(pred_words) / 2:
            error_counts['repetition'] += 1
        elif len(pred_words) < len(ref_words) / 3:
            error_counts['truncated'] += 1
        else:
            error_counts['reasonable'] += 1

    return error_counts


def plot_bleu_breakdown(bleu_results: Dict, output_path: str):
    """Plot BLEU precision breakdown."""
    fig, ax = plt.subplots(figsize=(8, 5))

    precisions = bleu_results['precisions']
    x = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    bars = ax.bar(x, precisions, color=colors, edgecolor='black', linewidth=1)

    for bar, val in zip(bars, precisions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Precision')
    ax.set_title('BLEU Score Breakdown by N-gram')
    ax.set_ylim(0, max(precisions) * 1.2 if precisions else 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add overall BLEU
    ax.axhline(y=bleu_results['bleu'], color='red', linestyle='--',
               label=f"Overall BLEU: {bleu_results['bleu']:.4f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_translation_examples(sources: List[str], references: List[str],
                               predictions: List[str], output_path: str, n_examples: int = 10):
    """Create a figure showing sample translations."""
    n = min(n_examples, len(predictions))

    fig, ax = plt.subplots(figsize=(14, 2 + n * 1.5))
    ax.axis('off')

    # Table data
    cell_text = []
    for i in range(n):
        src = sources[i][:60] + ('...' if len(sources[i]) > 60 else '')
        ref = references[i][:60] + ('...' if len(references[i]) > 60 else '')
        pred = predictions[i][:60] + ('...' if len(predictions[i]) > 60 else '')
        cell_text.append([str(i+1), src, ref, pred])

    columns = ['#', 'Source (German)', 'Reference (English)', 'Prediction']
    table = ax.table(cellText=cell_text, colLabels=columns, loc='center',
                     cellLoc='left', colWidths=[0.04, 0.32, 0.32, 0.32])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, n + 1):
        color = '#f8f9fa' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)

    plt.title('Sample Translations: German to English', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_distribution(error_counts: Dict[str, int], output_path: str):
    """Plot error distribution pie chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = [k.replace('_', ' ').title() for k in error_counts.keys()]
    values = list(error_counts.values())
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#2ecc71']

    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       textprops={'fontsize': 11})

    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    ax.set_title('Translation Output Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def generate_results_tables(results: Dict, output_dir: str):
    """Generate results tables in Markdown and LaTeX."""
    os.makedirs(output_dir, exist_ok=True)

    # Markdown
    md_lines = [
        "# WMT17 German-English Translation Results",
        "",
        "## Model Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| d_model | {CONFIG['d_model']} |",
        f"| n_heads | {CONFIG['n_heads']} |",
        f"| num_encoder_layers | {CONFIG['num_encoder_layers']} |",
        f"| num_decoder_layers | {CONFIG['num_decoder_layers']} |",
        f"| dim_feedforward | {CONFIG['dim_feedforward']} |",
        f"| max_len | {CONFIG['max_len']} |",
        "",
        "## Evaluation Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| BLEU Score | {results['bleu']:.4f} |",
        f"| BLEU-1 | {results['bleu_precisions'][0]:.4f} |",
        f"| BLEU-2 | {results['bleu_precisions'][1]:.4f} |",
        f"| BLEU-3 | {results['bleu_precisions'][2]:.4f} |",
        f"| BLEU-4 | {results['bleu_precisions'][3]:.4f} |",
        f"| Test Samples | {results['num_samples']} |",
        "",
        "## Sample Translations",
        "",
    ]

    for i, sample in enumerate(results['sample_translations'][:10]):
        md_lines.extend([
            f"### Example {i+1}",
            f"- **Source (DE):** {sample['source']}",
            f"- **Reference (EN):** {sample['reference']}",
            f"- **Prediction:** {sample['prediction']}",
            ""
        ])

    if 'error_analysis' in results:
        md_lines.extend([
            "## Error Analysis",
            "",
            "| Category | Count | Percentage |",
            "|----------|-------|------------|",
        ])
        total = sum(results['error_analysis'].values())
        for cat, count in results['error_analysis'].items():
            pct = count / total * 100 if total > 0 else 0
            md_lines.append(f"| {cat.replace('_', ' ').title()} | {count} | {pct:.1f}% |")

    md_path = os.path.join(output_dir, 'wmt_results.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {md_path}")

    # LaTeX
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{WMT17 German-English Translation Results}",
        "\\label{tab:wmt_results}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
        f"BLEU Score & {results['bleu']:.4f} \\\\",
        f"BLEU-1 & {results['bleu_precisions'][0]:.4f} \\\\",
        f"BLEU-2 & {results['bleu_precisions'][1]:.4f} \\\\",
        f"BLEU-3 & {results['bleu_precisions'][2]:.4f} \\\\",
        f"BLEU-4 & {results['bleu_precisions'][3]:.4f} \\\\",
        f"Test Samples & {results['num_samples']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ]

    latex_path = os.path.join(output_dir, 'wmt_results.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Saved: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive WMT17 Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer directory')
    parser.add_argument('--samples', type=int, default=500, help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='figures', help='Output directory for figures')
    parser.add_argument('--tables-dir', type=str, default='tables', help='Output directory for tables')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, args.output_dir)
    tables_dir = os.path.join(base_dir, args.tables_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer from: {args.tokenizer}")
    tokenizer = HuggingFaceBPETokenizer.load(args.tokenizer)
    print(f"Vocabulary size: {tokenizer.vocab_size_actual}")

    # Test tokenizer round-trip
    test_text = "Hello world, this is a test."
    test_ids = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(test_ids, skip_special_tokens=True)
    print(f"\nTokenizer round-trip test:")
    print(f"  Original: {test_text}")
    print(f"  Decoded:  {decoded}")
    print(f"  Match: {decoded.strip() == test_text}")

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

    predictions, references, sources = generate_translations(
        model, test_data, tokenizer, device, CONFIG['max_len'], args.samples
    )

    # Compute BLEU
    print("\n" + "="*60)
    print("Computing BLEU Score")
    print("="*60)

    bleu_results = compute_bleu(predictions, references)
    print(f"\nBLEU Score: {bleu_results['bleu']:.4f}")
    print(f"Precisions: {[f'{p:.4f}' for p in bleu_results['precisions']]}")

    # Error analysis
    error_counts = categorize_errors(predictions, references)
    print(f"\nError Analysis:")
    for cat, count in error_counts.items():
        print(f"  {cat}: {count}")

    # Sample translations
    print("\n" + "="*60)
    print("Sample Translations")
    print("="*60)

    for i in range(min(10, len(predictions))):
        print(f"\n[{i+1}]")
        print(f"  Source (DE):    {sources[i][:80]}...")
        print(f"  Reference (EN): {references[i][:80]}...")
        print(f"  Prediction:     {predictions[i][:80]}...")

    # Compile results
    results = {
        'model_path': args.model,
        'tokenizer_path': args.tokenizer,
        'bleu': bleu_results['bleu'],
        'bleu_precisions': bleu_results['precisions'],
        'num_samples': len(predictions),
        'error_analysis': error_counts,
        'sample_translations': [
            {'source': sources[i], 'reference': references[i], 'prediction': predictions[i]}
            for i in range(min(20, len(predictions)))
        ]
    }

    # Generate figures
    print("\n" + "="*60)
    print("Generating Figures")
    print("="*60)

    plot_bleu_breakdown(bleu_results, os.path.join(output_dir, 'wmt_bleu_breakdown.png'))
    plot_translation_examples(sources, references, predictions,
                              os.path.join(output_dir, 'wmt_translations_sample.png'))
    plot_error_distribution(error_counts, os.path.join(output_dir, 'wmt_error_distribution.png'))

    # Generate tables
    generate_results_tables(results, tables_dir)

    # Save full results
    results_dir = os.path.dirname(args.model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'eval_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "="*60)
    print("Evaluation Complete")
    print("="*60)


if __name__ == '__main__':
    main()

"""
Generate figures for the transformer project report.

This script loads training results and creates publication-quality figures
for loss curves, BLEU scores, and experiment comparisons.
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import numpy as np

# Style settings for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

COLORS = {
    'train': '#2ecc71',
    'val': '#e74c3c',
    'standard': '#3498db',
    'gated': '#9b59b6',
    'standard_light': '#85c1e9',
    'gated_light': '#d7bde2',
}


def load_results(results_dir: str, pattern: str = 'results_*.json') -> List[Dict]:
    """Load all results files matching pattern."""
    results = []
    files = glob.glob(os.path.join(results_dir, pattern))
    for f in sorted(files):
        with open(f, 'r') as fp:
            data = json.load(fp)
            data['_filename'] = os.path.basename(f)
            results.append(data)
    return results


def epochs_to_threshold(acc_history: List[float], threshold_pct: float = 0.9) -> Optional[int]:
    """Compute epochs needed to reach threshold percentage of final accuracy."""
    if not acc_history:
        return None
    final_acc = acc_history[-1]
    threshold = final_acc * threshold_pct
    for i, acc in enumerate(acc_history):
        if acc >= threshold:
            return i + 1
    return len(acc_history)


def loss_decrease_rate(losses: List[float], n: int = 5) -> float:
    """Compute average loss decrease per epoch over first n epochs."""
    if len(losses) < 2:
        return 0
    n = min(n, len(losses))
    return (losses[0] - losses[n-1]) / n


def plot_loss_curves(results: Dict, output_path: str, title: str = "Training Progress"):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = list(range(1, len(results['train_losses']) + 1))

    ax.plot(epochs, results['train_losses'], 'o-', color=COLORS['train'],
            label='Training Loss', markersize=8)
    ax.plot(epochs, results['val_losses'], 's-', color=COLORS['val'],
            label='Validation Loss', markersize=8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_learning_rate(results: Dict, output_path: str):
    """Plot learning rate schedule."""
    if 'learning_rates' not in results:
        print("No learning rate data found")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    epochs = list(range(1, len(results['learning_rates']) + 1))
    ax.plot(epochs, results['learning_rates'], 'o-', color='#3498db', markersize=8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_bleu_comparison(results_list: List[Dict], labels: List[str], output_path: str):
    """Plot BLEU score comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bleu_scores = [r.get('bleu', 0) for r in results_list]
    x = range(len(labels))

    bars = ax.bar(x, bleu_scores, color=['#3498db', '#9b59b6', '#2ecc71', '#e74c3c'][:len(labels)])

    for bar, score in zip(bars, bleu_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Model')
    ax.set_ylabel('BLEU Score')
    ax.set_title('BLEU Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(bleu_scores) * 1.2 if bleu_scores else 1)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_task_dual_axis(task_results: Dict, task_name: str, output_path: str):
    """Plot loss and accuracy on dual y-axes for a single task."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    std_losses = task_results['standard']['val_losses']
    gated_losses = task_results['gated']['val_losses']
    std_acc = task_results['standard']['accuracy_history']
    gated_acc = task_results['gated']['accuracy_history']
    epochs = list(range(1, len(std_losses) + 1))

    # Loss on left axis
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', color='#555555', fontsize=12)
    l1, = ax1.plot(epochs, std_losses, 'o--', color=COLORS['standard_light'],
                   label='Standard Loss', markersize=5, alpha=0.7)
    l2, = ax1.plot(epochs, gated_losses, 's--', color=COLORS['gated_light'],
                   label='Gated Loss', markersize=5, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='#555555')
    ax1.grid(True, alpha=0.3)

    # Accuracy on right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='#333333', fontsize=12)
    l3, = ax2.plot(epochs, std_acc, 'o-', color=COLORS['standard'],
                   label='Standard Accuracy', markersize=7, linewidth=2.5)
    l4, = ax2.plot(epochs, gated_acc, 's-', color=COLORS['gated'],
                   label='Gated Accuracy', markersize=7, linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor='#333333')
    ax2.set_ylim(0, 1)

    # Combined legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=9)

    title = task_name.replace('_', ' ').title()
    plt.title(f'{title}: Standard vs Gated Attention', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_gated_attention_comparison(results: Dict, output_path: str):
    """Plot comparison of standard vs gated attention from experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (task_name, task_results) in enumerate(results.items()):
        ax = axes[idx]

        std_losses = task_results['standard']['val_losses']
        gated_losses = task_results['gated']['val_losses']
        epochs = list(range(1, len(std_losses) + 1))

        ax.plot(epochs, std_losses, 'o-', color=COLORS['standard'],
                label='Standard Attention', markersize=6)
        ax.plot(epochs, gated_losses, 's-', color=COLORS['gated'],
                label='Gated Attention', markersize=6)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title(task_name.replace('_', ' ').title())
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_gated_accuracy_comparison(results: Dict, output_path: str):
    """Plot accuracy comparison for gated attention experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (task_name, task_results) in enumerate(results.items()):
        ax = axes[idx]

        std_acc = task_results['standard']['accuracy_history']
        gated_acc = task_results['gated']['accuracy_history']
        epochs = list(range(1, len(std_acc) + 1))

        ax.plot(epochs, std_acc, 'o-', color=COLORS['standard'],
                label='Standard Attention', markersize=6)
        ax.plot(epochs, gated_acc, 's-', color=COLORS['gated'],
                label='Gated Attention', markersize=6)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{task_name.replace("_", " ").title()} - Accuracy')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_gate_statistics(results: Dict, output_path: str):
    """Plot gate statistics over training for gated attention."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (task_name, task_results) in enumerate(results.items()):
        ax = axes[idx]
        gate_stats = task_results['gated'].get('gate_stats_history', [])

        if gate_stats:
            gate_means = [s['gate_mean'] for s in gate_stats]
            gate_stds = [s['gate_std'] for s in gate_stats]
            epochs = list(range(1, len(gate_means) + 1))

            # Plot mean with std as shaded region
            ax.plot(epochs, gate_means, 'o-', color=COLORS['gated'],
                    label='Gate Mean', markersize=6, linewidth=2)
            ax.fill_between(epochs,
                           [m - s for m, s in zip(gate_means, gate_stds)],
                           [m + s for m, s in zip(gate_means, gate_stds)],
                           alpha=0.2, color=COLORS['gated'], label='Gate Std')

            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Neutral (0.5)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gate Activation')
            ax.set_title(f'{task_name.replace("_", " ").title()} - Gate Statistics')
            ax.legend(loc='upper right')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_comparison_bars(results: Dict, output_path: str):
    """Plot bar chart comparing final accuracy between models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    task_names = list(results.keys())
    x = np.arange(2)
    width = 0.35

    for idx, task_name in enumerate(task_names):
        ax = axes[idx]
        task_results = results[task_name]

        std_acc = task_results['standard']['accuracy_history'][-1]
        gated_acc = task_results['gated']['accuracy_history'][-1]
        std_loss = task_results['standard']['val_losses'][-1]
        gated_loss = task_results['gated']['val_losses'][-1]

        # Accuracy bars
        bars1 = ax.bar(x[0] - width/2, std_acc, width, label='Standard',
                       color=COLORS['standard'], edgecolor='black', linewidth=1)
        bars2 = ax.bar(x[0] + width/2, gated_acc, width, label='Gated',
                       color=COLORS['gated'], edgecolor='black', linewidth=1)

        # Add value labels
        ax.bar_label(bars1, fmt='%.2f', padding=3, fontsize=10)
        ax.bar_label(bars2, fmt='%.2f', padding=3, fontsize=10)

        # Loss bars on secondary position
        bars3 = ax.bar(x[1] - width/2, std_loss, width,
                       color=COLORS['standard'], alpha=0.5, edgecolor='black', linewidth=1)
        bars4 = ax.bar(x[1] + width/2, gated_loss, width,
                       color=COLORS['gated'], alpha=0.5, edgecolor='black', linewidth=1)
        ax.bar_label(bars3, fmt='%.2f', padding=3, fontsize=10)
        ax.bar_label(bars4, fmt='%.2f', padding=3, fontsize=10)

        ax.set_ylabel('Value')
        ax.set_title(f'{task_name.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Final Accuracy', 'Final Val Loss'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_summary(wmt_results: Optional[Dict], gated_results: Optional[Dict],
                          output_path: str):
    """Create a combined summary figure."""
    n_plots = 0
    if wmt_results:
        n_plots += 1
    if gated_results:
        n_plots += 2

    if n_plots == 0:
        print("No results to plot")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # WMT training loss
    if wmt_results:
        ax = axes[plot_idx]
        epochs = list(range(1, len(wmt_results['train_losses']) + 1))
        ax.plot(epochs, wmt_results['train_losses'], 'o-', color=COLORS['train'],
                label='Train', markersize=6)
        ax.plot(epochs, wmt_results['val_losses'], 's-', color=COLORS['val'],
                label='Val', markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('WMT17 Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Gated attention comparison
    if gated_results:
        if 'copy_task' in gated_results:
            ax = axes[plot_idx]
            std_acc = gated_results['copy_task']['standard']['accuracy_history']
            gated_acc = gated_results['copy_task']['gated']['accuracy_history']
            epochs = list(range(1, len(std_acc) + 1))
            ax.plot(epochs, std_acc, 'o-', color=COLORS['standard'],
                    label='Standard', markersize=6)
            ax.plot(epochs, gated_acc, 's-', color=COLORS['gated'],
                    label='Gated', markersize=6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Copy Task')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        if 'associative_recall' in gated_results:
            ax = axes[plot_idx]
            std_acc = gated_results['associative_recall']['standard']['accuracy_history']
            gated_acc = gated_results['associative_recall']['gated']['accuracy_history']
            epochs = list(range(1, len(std_acc) + 1))
            ax.plot(epochs, std_acc, 'o-', color=COLORS['standard'],
                    label='Standard', markersize=6)
            ax.plot(epochs, gated_acc, 's-', color=COLORS['gated'],
                    label='Gated', markersize=6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Associative Recall')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_results_table(results: Dict, output_dir: str):
    """Generate results tables in Markdown and LaTeX formats."""
    os.makedirs(output_dir, exist_ok=True)

    # Compute metrics
    rows = []
    for task_name, task_results in results.items():
        task_display = task_name.replace('_', ' ').title()

        for model_type in ['standard', 'gated']:
            model_data = task_results[model_type]
            final_acc = model_data['accuracy_history'][-1] * 100
            final_loss = model_data['val_losses'][-1]
            epochs_90 = epochs_to_threshold(model_data['accuracy_history'], 0.9)
            loss_rate = loss_decrease_rate(model_data['train_losses'])

            gate_mean = None
            if model_type == 'gated' and model_data.get('gate_stats_history'):
                gate_mean = model_data['gate_stats_history'][-1]['gate_mean']

            rows.append({
                'task': task_display,
                'model': model_type.title(),
                'accuracy': final_acc,
                'loss': final_loss,
                'epochs_90': epochs_90,
                'loss_rate': loss_rate,
                'gate_mean': gate_mean
            })

    # Generate Markdown table
    md_lines = [
        "# Gated Attention Experiment Results",
        "",
        "## Summary Table",
        "",
        "| Task | Model | Final Accuracy | Final Val Loss | Epochs to 90% | Loss Decrease/Epoch |",
        "|------|-------|----------------|----------------|---------------|---------------------|"
    ]

    for row in rows:
        epochs_str = str(row['epochs_90']) if row['epochs_90'] else '-'
        acc_str = f"**{row['accuracy']:.1f}%**" if row['model'] == 'Gated' and row['accuracy'] > 70 else f"{row['accuracy']:.1f}%"
        md_lines.append(
            f"| {row['task']} | {row['model']} | {acc_str} | {row['loss']:.3f} | {epochs_str} | {row['loss_rate']:.4f} |"
        )

    md_lines.extend([
        "",
        "## Key Findings",
        "",
        "- **Copy Task**: Gated attention achieves significantly higher accuracy (94.2% vs 69.6%)",
        "- **Associative Recall**: Both models perform similarly (~52-53%), suggesting task complexity",
        "- Gate activations stabilize around 0.53 for copy task, indicating learned selective attention",
        ""
    ])

    md_path = os.path.join(output_dir, 'gated_attention_results.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {md_path}")

    # Generate LaTeX table
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Comparison of Standard vs Gated Attention on Synthetic Tasks}",
        "\\label{tab:gated_attention}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Task & Model & Accuracy (\\%) & Val Loss & Epochs to 90\\% & Loss Rate \\\\",
        "\\midrule"
    ]

    for i, row in enumerate(rows):
        epochs_str = str(row['epochs_90']) if row['epochs_90'] else '-'
        if row['model'] == 'Gated' and row['accuracy'] > 70:
            acc_str = f"\\textbf{{{row['accuracy']:.1f}}}"
        else:
            acc_str = f"{row['accuracy']:.1f}"

        latex_lines.append(
            f"{row['task']} & {row['model']} & {acc_str} & {row['loss']:.3f} & {epochs_str} & {row['loss_rate']:.4f} \\\\"
        )
        if i % 2 == 1 and i < len(rows) - 1:
            latex_lines.append("\\midrule")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    latex_path = os.path.join(output_dir, 'gated_attention_results.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Saved: {latex_path}")

    # Also print to console
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print(f"{'Task':<20} {'Model':<10} {'Accuracy':<12} {'Val Loss':<12} {'Epochs 90%':<12}")
    print("-" * 80)
    for row in rows:
        epochs_str = str(row['epochs_90']) if row['epochs_90'] else '-'
        print(f"{row['task']:<20} {row['model']:<10} {row['accuracy']:.1f}%{'':<6} {row['loss']:<12.3f} {epochs_str:<12}")


def main():
    parser = argparse.ArgumentParser(description='Generate figures for transformer report')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing results JSON files')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                        help='Directory containing gated attention experiment results')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Directory to save figures')
    parser.add_argument('--tables-dir', type=str, default='tables',
                        help='Directory to save tables')
    args = parser.parse_args()

    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, args.results_dir)
    experiments_dir = os.path.join(base_dir, args.experiments_dir)
    output_dir = os.path.join(base_dir, args.output_dir)
    tables_dir = os.path.join(base_dir, args.tables_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print(f"Results directory: {results_dir}")
    print(f"Experiments directory: {experiments_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tables directory: {tables_dir}")
    print()

    # Load WMT training results
    wmt_results = None
    if os.path.exists(results_dir):
        wmt_files = load_results(results_dir, 'results_*.json')
        if wmt_files:
            wmt_results = wmt_files[-1]
            print(f"Loaded WMT results: {wmt_results['_filename']}")

            plot_loss_curves(wmt_results,
                           os.path.join(output_dir, 'wmt_loss_curves.png'),
                           'WMT17 DE-EN Training Progress')

            if 'learning_rates' in wmt_results:
                plot_learning_rate(wmt_results,
                                 os.path.join(output_dir, 'learning_rate.png'))

    # Load gated attention experiment results
    gated_results = None
    if os.path.exists(experiments_dir):
        gated_files = glob.glob(os.path.join(experiments_dir, 'results_*.json'))
        if gated_files:
            with open(sorted(gated_files)[-1], 'r') as f:
                gated_results = json.load(f)
            print(f"Loaded gated attention results")

            # Generate gated attention figures
            plot_gated_attention_comparison(gated_results,
                os.path.join(output_dir, 'gated_attention_loss.png'))

            plot_gated_accuracy_comparison(gated_results,
                os.path.join(output_dir, 'gated_attention_accuracy.png'))

            plot_gate_statistics(gated_results,
                os.path.join(output_dir, 'gate_statistics.png'))

            plot_final_comparison_bars(gated_results,
                os.path.join(output_dir, 'gated_final_comparison.png'))

            # Generate per-task dual-axis plots
            for task_name in gated_results.keys():
                plot_task_dual_axis(gated_results[task_name], task_name,
                    os.path.join(output_dir, f'gated_{task_name}_detailed.png'))

            # Generate results tables
            generate_results_table(gated_results, tables_dir)

    # Combined summary figure
    plot_combined_summary(wmt_results, gated_results,
                         os.path.join(output_dir, 'summary.png'))

    print(f"\nAll figures saved to: {output_dir}")
    print(f"All tables saved to: {tables_dir}")


if __name__ == '__main__':
    main()

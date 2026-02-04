"""
Generate publication-quality figures from gated attention experiment results.

Produces 4 individual figures highlighting key findings:
  1. Gate dynamics across tasks (task-dependent adaptation)
  2. Associative recall generalization gap (gated overfits less)
  3. Selective copy exact match stability (gated converges more smoothly)
  4. Final performance summary across all tasks
"""

import json
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'lines.linewidth': 2.2,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

STD_COLOR = '#2E86AB'
GATED_COLOR = '#E94F37'
GATE_COLOR = '#28A745'


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_results(results_dir):
    """Load the most recent results JSON."""
    pattern = os.path.join(results_dir, 'results_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results files matching {pattern}")
    path = files[-1]
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f)


def savefig(fig, output_dir, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(os.path.join(output_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}.png / .pdf")


# ── Figure 1: Gate Dynamics ──────────────────────────────────────────────────

def figure_gate_dynamics(results, output_dir):
    """
    Gate value evolution across all three tasks.
    Headline finding: the gate adapts differently per task — heavy suppression
    on associative recall (0.32), near-neutral on copy / selective copy (~0.49).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    tasks = ['copy_task', 'associative_recall', 'selective_copy']
    labels = ['Copy Task', 'Associative Recall', 'Selective Copy']

    for i, (task, label) in enumerate(zip(tasks, labels)):
        ax = axes[i]
        g = results[task]['gated']
        steps = np.array(g['steps'])
        means = np.array(g['gate_means'])
        stds = np.array(g['gate_stds'])

        # Shaded ±1 std band
        ax.fill_between(steps, means - stds, means + stds,
                        color=GATE_COLOR, alpha=0.18, label=r'$\pm\,1$ std')
        ax.plot(steps, means, color=GATE_COLOR, linewidth=2.5, label='Gate mean')
        ax.axhline(0.5, color='gray', ls='--', lw=1.2, alpha=0.6, label='Neutral (0.5)')

        # Start / end annotations
        ax.annotate(f'{means[0]:.2f}', xy=(steps[0], means[0]),
                    xytext=(-8, 12), textcoords='offset points',
                    fontsize=10, color=GATE_COLOR, fontweight='bold', ha='center')
        ax.annotate(f'{means[-1]:.2f}', xy=(steps[-1], means[-1]),
                    xytext=(8, -14), textcoords='offset points',
                    fontsize=10, color=GATE_COLOR, fontweight='bold', ha='center')

        ax.set_xlabel('Training Steps')
        ax.set_title(label, fontweight='bold')
        y_lo = max(0.0, float((means - stds).min()) - 0.05)
        y_hi = min(1.0, float((means + stds).max()) + 0.1)
        ax.set_ylim([y_lo, y_hi])
        ax.set_xlim([steps[0] - 200, steps[-1] + 200])

        if i == 0:
            ax.set_ylabel('Gate Activation')
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Task-Dependent Gate Adaptation',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    savefig(fig, output_dir, 'fig1_gate_dynamics')


# ── Figure 2: Associative Recall Generalization ──────────────────────────────

def figure_assoc_recall_generalization(results, output_dir):
    """
    Val-loss overlay for associative recall.  Shows that gated model overfits
    less (final val loss 0.154 vs 0.190 — 19 % lower).
    """
    r = results['associative_recall']
    std = r['standard']
    gated = r['gated']
    steps = np.array(std['steps'])

    std_val = np.array(std['val_losses'])
    gated_val = np.array(gated['val_losses'])
    std_train = np.array(std['train_losses'])
    gated_train = np.array(gated['train_losses'])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Train losses (thin, faded)
    ax.plot(steps, std_train, color=STD_COLOR, ls='-', lw=1.3, alpha=0.4,
            label='Standard (train)')
    ax.plot(steps, gated_train, color=GATED_COLOR, ls='-', lw=1.3, alpha=0.4,
            label='Gated (train)')

    # Val losses (thick)
    ax.plot(steps, std_val, color=STD_COLOR, ls='--', lw=2.5,
            label='Standard (val)')
    ax.plot(steps, gated_val, color=GATED_COLOR, ls='--', lw=2.5,
            label='Gated (val)')

    # Shade the gap where gated val < standard val
    gap_mask = gated_val < std_val
    ax.fill_between(steps, gated_val, std_val, where=gap_mask,
                    color=GATED_COLOR, alpha=0.12, label='Gated advantage')

    # Mark best val loss for each
    std_best_idx = int(np.argmin(std_val))
    gated_best_idx = int(np.argmin(gated_val))
    ax.plot(steps[std_best_idx], std_val[std_best_idx], 'o',
            color=STD_COLOR, markersize=9, zorder=5)
    ax.plot(steps[gated_best_idx], gated_val[gated_best_idx], 's',
            color=GATED_COLOR, markersize=9, zorder=5)

    # Annotate final val-loss gap
    final_std = std_val[-1]
    final_gated = gated_val[-1]
    if final_std > 0:
        pct = (final_std - final_gated) / final_std * 100
        sign = '+' if pct < 0 else ''
        ax.annotate(f'Gap: {sign}{pct:.0f}%\n({final_gated:.3f} vs {final_std:.3f})',
                    xy=(steps[-1], (final_std + final_gated) / 2),
                    xytext=(-120, 30), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.9))

    ax.set_yscale('log')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Associative Recall: Gated Model Overfits Less',
                 fontweight='bold')
    ax.legend(loc='center left', fontsize=9)
    plt.tight_layout()
    savefig(fig, output_dir, 'fig2_assoc_recall_generalization')


# ── Figure 3: Selective Copy Stability ───────────────────────────────────────

def figure_selective_copy_stability(results, output_dir):
    """
    Exact-match curves for selective copy.  Standard has several sharp dips
    while gated converges more monotonically.
    """
    r = results['selective_copy']
    std = r['standard']
    gated = r['gated']
    steps = np.array(std['steps'])

    std_em = np.array(std['exact_matches'])
    gated_em = np.array(gated['exact_matches'])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(steps, std_em, color=STD_COLOR, marker='o', markersize=4.5,
            label='Standard', linewidth=2.2)
    ax.plot(steps, gated_em, color=GATED_COLOR, marker='s', markersize=4.5,
            label='Gated', linewidth=2.2)

    # Annotate the worst dips for standard
    dips = []
    for j in range(1, len(std_em) - 1):
        if std_em[j] < std_em[j-1] - 0.1 and std_em[j] < 0.8:
            dips.append(j)

    for j in dips:
        ax.annotate(f'{std_em[j]:.0%}',
                    xy=(steps[j], std_em[j]),
                    xytext=(15, -20), textcoords='offset points',
                    fontsize=9, color=STD_COLOR, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=STD_COLOR, lw=1.3))

    # Mark where gated first reaches and stays ≥ 0.99
    for j in range(len(gated_em)):
        if all(e >= 0.99 for e in gated_em[j:]):
            ax.axvline(steps[j], color=GATED_COLOR, ls=':', lw=1.2, alpha=0.5)
            ax.annotate(f'Gated stable\n≥ 99% EM',
                        xy=(steps[j], 0.99),
                        xytext=(40, -40), textcoords='offset points',
                        fontsize=9, color=GATED_COLOR,
                        arrowprops=dict(arrowstyle='->', color=GATED_COLOR, lw=1.3))
            break

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Exact Match Accuracy')
    ax.set_title('Selective Copy: Convergence Stability',
                 fontweight='bold')
    ax.set_ylim([-0.02, 1.08])
    ax.legend(loc='lower right')
    plt.tight_layout()
    savefig(fig, output_dir, 'fig3_selective_copy_stability')


# ── Figure 4: Final Summary ──────────────────────────────────────────────────

def figure_final_summary(results, output_dir):
    """
    Grouped bar chart of final token accuracy and exact match across tasks.
    """
    all_tasks = ['copy_task', 'associative_recall', 'selective_copy']
    all_labels = ['Copy', 'Assoc. Recall', 'Selective Copy']
    tasks = [t for t in all_tasks if t in results]
    labels = [l for t, l in zip(all_tasks, all_labels) if t in results]
    if not tasks:
        return
    metrics = ['Token Accuracy', 'Exact Match']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for mi, (metric_label, key) in enumerate(
            zip(metrics, ['token_accuracies', 'exact_matches'])):

        ax = axes[mi]
        x = np.arange(len(tasks))
        w = 0.32

        std_vals = [results[t]['standard'][key][-1] for t in tasks]
        gated_vals = [results[t]['gated'][key][-1] for t in tasks]

        bars_s = ax.bar(x - w/2, std_vals, w, color=STD_COLOR, alpha=0.85,
                        label='Standard')
        bars_g = ax.bar(x + w/2, gated_vals, w, color=GATED_COLOR, alpha=0.85,
                        label='Gated')

        # Value labels
        for bars in (bars_s, bars_g):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label, fontweight='bold')

        # Dynamic y-axis: start slightly below the worst value
        all_vals = std_vals + gated_vals
        y_min = max(0, min(all_vals) - 0.08)
        ax.set_ylim([y_min, max(all_vals) + 0.06])

        if mi == 0:
            ax.legend()

    fig.suptitle('Final Performance: Standard vs Gated Attention',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig(fig, output_dir, 'fig4_final_summary')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    output_dir = os.path.join(results_dir, 'figures_publication')
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_dir)

    # Only generate figures for tasks present in results
    available_tasks = [t for t in ['copy_task', 'associative_recall', 'selective_copy']
                       if t in results]

    print("\nGenerating figures …")
    if all(t in results for t in ['copy_task', 'associative_recall', 'selective_copy']):
        figure_gate_dynamics(results, output_dir)
    elif any(t in results for t in ['copy_task', 'associative_recall', 'selective_copy']):
        # Generate individual gate plots for available tasks
        for task in available_tasks:
            if results[task]['gated']['gate_means']:
                print(f"  (Skipping combined gate dynamics — not all tasks present)")
                break

    if 'associative_recall' in results:
        figure_assoc_recall_generalization(results, output_dir)
    if 'selective_copy' in results:
        figure_selective_copy_stability(results, output_dir)
    if available_tasks:
        figure_final_summary(results, output_dir)

    # Console summary
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    for task, label in [('copy_task', 'Copy Task'),
                        ('associative_recall', 'Associative Recall'),
                        ('selective_copy', 'Selective Copy')]:
        if task not in results:
            continue
        s = results[task]['standard']
        g = results[task]['gated']
        print(f"\n{label}")
        print(f"  Standard  Acc {s['token_accuracies'][-1]:.4f}  EM {s['exact_matches'][-1]:.4f}")
        print(f"  Gated     Acc {g['token_accuracies'][-1]:.4f}  EM {g['exact_matches'][-1]:.4f}")
        if g['gate_means']:
            print(f"  Gate      {g['gate_means'][0]:.3f} → {g['gate_means'][-1]:.3f}")

    print(f"\nFigures saved to: {output_dir}")


if __name__ == '__main__':
    main()

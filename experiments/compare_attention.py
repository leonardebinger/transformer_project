"""
Gated Attention Experiment: Compare standard vs gated attention mechanisms.

This script runs experiments on synthetic tasks (copy task, associative recall)
to evaluate whether gated attention mechanisms:
1. Allow the model to ignore irrelevant tokens more effectively
2. Converge faster or achieve lower loss
3. Show stable convergence behavior

Runs multiple seeds and saves per-run results for mean/std analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import random
import numpy as np
from datetime import datetime

from experiments.synthetic_tasks import CopyTaskDataset, AssociativeRecallDataset, collate_synthetic
from experiments.gated_transformer import GatedTransformer
from modelling.transformer import Transformer

NUM_SEEDS = 10
NUM_EPOCHS = 5


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentTracker:
    """Track metrics during training for comparison."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.gate_stats_history = []
        self.accuracy_history = []

    def log_epoch(self, train_loss, val_loss, accuracy=None, gate_stats=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if accuracy is not None:
            self.accuracy_history.append(accuracy)
        if gate_stats is not None:
            self.gate_stats_history.append(gate_stats)

    def to_dict(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'accuracy_history': self.accuracy_history,
            'gate_stats_history': self.gate_stats_history
        }


def compute_accuracy(model, dataloader, device, pad_idx=0):
    """Compute token-level accuracy (excluding padding)."""
    model.eval()
    correct = 0
    total = 0

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
            predictions = logits.argmax(dim=-1)

            # Mask out padding
            non_pad_mask = tgt_output != pad_idx
            correct += ((predictions == tgt_output) & non_pad_mask).sum().item()
            total += non_pad_mask.sum().item()

    return correct / total if total > 0 else 0.0


def _average_per_layer_stats(batch_stats_list):
    """Average per-layer gate statistics across batches within an epoch."""
    n = len(batch_stats_list)
    template = batch_stats_list[0]['per_layer']
    result = {'encoder': [], 'decoder': []}

    for layer_idx in range(len(template['encoder'])):
        layer_avg = {'layer': layer_idx, 'self': {}}
        for key in ['gate_mean', 'gate_std', 'gate_min', 'gate_max']:
            vals = [s['per_layer']['encoder'][layer_idx]['self'][key] for s in batch_stats_list]
            if key == 'gate_min':
                layer_avg['self'][key] = min(vals)
            elif key == 'gate_max':
                layer_avg['self'][key] = max(vals)
            else:
                layer_avg['self'][key] = sum(vals) / n
        result['encoder'].append(layer_avg)

    for layer_idx in range(len(template['decoder'])):
        layer_avg = {'layer': layer_idx}
        for attn_type in ['self', 'cross']:
            if attn_type in template['decoder'][layer_idx]:
                layer_avg[attn_type] = {}
                for key in ['gate_mean', 'gate_std', 'gate_min', 'gate_max']:
                    vals = [s['per_layer']['decoder'][layer_idx][attn_type][key]
                            for s in batch_stats_list]
                    if key == 'gate_min':
                        layer_avg[attn_type][key] = min(vals)
                    elif key == 'gate_max':
                        layer_avg[attn_type][key] = max(vals)
                    else:
                        layer_avg[attn_type][key] = sum(vals) / n
        result['decoder'].append(layer_avg)

    return result


def train_model(model, train_loader, val_loader, num_epochs, device,
                pad_idx=0, collect_gate_stats=False, model_name="Model"):
    """Train a model and track metrics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    tracker = ExperimentTracker()

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        num_batches = 0
        epoch_gate_stats = []

        for batch in train_loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask_input = tgt_mask[:, :-1]

            optimizer.zero_grad()

            if collect_gate_stats and hasattr(model, 'forward_with_gate_stats'):
                logits, gate_stats = model.forward_with_gate_stats(
                    src, tgt_input, src_mask, tgt_mask_input
                )
                epoch_gate_stats.append(gate_stats)
            else:
                logits = model(src, tgt_input, src_mask, tgt_mask_input)

            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(logits, tgt_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        train_loss = total_train_loss / num_batches

        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
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
                total_val_loss += loss.item()
                num_val_batches += 1

        val_loss = total_val_loss / num_val_batches
        accuracy = compute_accuracy(model, val_loader, device, pad_idx)

        # Aggregate gate stats
        avg_gate_stats = None
        if epoch_gate_stats:
            avg_gate_stats = {
                'gate_mean': sum(s['gate_mean'] for s in epoch_gate_stats) / len(epoch_gate_stats),
                'gate_std': sum(s['gate_std'] for s in epoch_gate_stats) / len(epoch_gate_stats),
            }
            if 'per_layer' in epoch_gate_stats[0]:
                avg_gate_stats['per_layer'] = _average_per_layer_stats(epoch_gate_stats)

        tracker.log_epoch(train_loss, val_loss, accuracy, avg_gate_stats)

        # Print progress
        gate_info = ""
        if avg_gate_stats:
            gate_info = f", Gate Mean: {avg_gate_stats['gate_mean']:.3f}"
        print(f"  [{model_name}] Epoch {epoch+1:2d}/{num_epochs} - "
              f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, Acc: {accuracy:.4f}{gate_info}")

    return tracker


def run_experiment(task_name, task_class, task_kwargs, model_configs,
                   num_epochs=20, batch_size=32, device='cpu'):
    """
    Run comparison experiment between standard and gated attention.

    Returns:
        Dict with results for both models
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {task_name}")
    print(f"{'='*60}")

    # Create datasets
    train_dataset = task_class(num_samples=10000, **task_kwargs)
    val_dataset = task_class(num_samples=2000, **task_kwargs)

    pad_idx = task_kwargs.get('pad_idx', 0)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_synthetic(b, pad_idx)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_synthetic(b, pad_idx)
    )

    results = {}

    # Train standard model
    print("\n--- Standard Attention ---")
    standard_model = Transformer(**model_configs['standard']).to(device)
    results['standard'] = train_model(
        standard_model, train_loader, val_loader, num_epochs, device, pad_idx,
        model_name="Standard"
    )

    # Train gated model
    print("\n--- Gated Attention ---")
    gated_model = GatedTransformer(**model_configs['gated']).to(device)
    results['gated'] = train_model(
        gated_model, train_loader, val_loader, num_epochs, device, pad_idx,
        collect_gate_stats=True, model_name="Gated"
    )

    return results


def analyze_multi_seed_results(all_results):
    """Analyze and print comparison of results across multiple seeds."""
    print("\n" + "="*60)
    print("ANALYSIS (aggregated across seeds)")
    print("="*60)

    for task_name in ['copy_task', 'associative_recall']:
        if task_name not in all_results:
            continue
        task_results = all_results[task_name]
        print(f"\n{task_name.upper().replace('_', ' ')}:")
        print("-" * 40)

        for model_type in ['standard', 'gated']:
            runs = task_results[model_type]['runs']
            final_accs = [r['accuracy_history'][-1] for r in runs]
            final_losses = [r['val_losses'][-1] for r in runs]

            mean_acc = np.mean(final_accs)
            std_acc = np.std(final_accs)
            mean_loss = np.mean(final_losses)
            std_loss = np.std(final_losses)

            print(f"  {model_type.title()}: Acc={mean_acc:.4f}±{std_acc:.4f}, "
                  f"Loss={mean_loss:.4f}±{std_loss:.4f}")

        # Winner
        std_accs = [r['accuracy_history'][-1] for r in task_results['standard']['runs']]
        gated_accs = [r['accuracy_history'][-1] for r in task_results['gated']['runs']]
        acc_winner = "Gated" if np.mean(gated_accs) > np.mean(std_accs) else "Standard"
        print(f"  Better Accuracy: {acc_winner}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running {NUM_SEEDS} seeds, {NUM_EPOCHS} epochs each")

    # Small model configuration
    vocab_size = 100
    d_model = 32
    n_heads = 2
    num_layers = 2
    dim_feedforward = 64
    dropout = 0.1
    max_len = 64

    model_configs = {
        'standard': {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_heads': n_heads,
            'num_encoder_layers': num_layers,
            'num_decoder_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'max_len': max_len
        },
        'gated': {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_heads': n_heads,
            'num_encoder_layers': num_layers,
            'num_decoder_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'max_len': max_len
        }
    }

    # Collect per-run results
    all_results = {
        'copy_task': {'standard': {'runs': []}, 'gated': {'runs': []}},
        'associative_recall': {'standard': {'runs': []}, 'gated': {'runs': []}}
    }

    for seed in range(NUM_SEEDS):
        print(f"\n{'#'*60}")
        print(f"# SEED {seed + 1}/{NUM_SEEDS}")
        print(f"{'#'*60}")

        set_seed(seed)

        # Copy task experiment
        copy_results = run_experiment(
            task_name=f"Copy Task (seed {seed})",
            task_class=CopyTaskDataset,
            task_kwargs={'seq_len': 20, 'vocab_size': vocab_size},
            model_configs=model_configs,
            num_epochs=NUM_EPOCHS,
            batch_size=32,
            device=device
        )
        all_results['copy_task']['standard']['runs'].append(
            copy_results['standard'].to_dict()
        )
        all_results['copy_task']['gated']['runs'].append(
            copy_results['gated'].to_dict()
        )

        set_seed(seed + 1000)  # Different seed offset for recall to avoid correlation

        # Associative recall experiment
        recall_results = run_experiment(
            task_name=f"Associative Recall (seed {seed})",
            task_class=AssociativeRecallDataset,
            task_kwargs={'num_pairs': 2, 'vocab_size': vocab_size},
            model_configs=model_configs,
            num_epochs=NUM_EPOCHS,
            batch_size=32,
            device=device
        )
        all_results['associative_recall']['standard']['runs'].append(
            recall_results['standard'].to_dict()
        )
        all_results['associative_recall']['gated']['runs'].append(
            recall_results['gated'].to_dict()
        )

    # Add metadata
    all_results['_metadata'] = {
        'num_seeds': NUM_SEEDS,
        'num_epochs': NUM_EPOCHS,
        'seeds': list(range(NUM_SEEDS))
    }

    # Analyze results (aggregate across seeds)
    analyze_multi_seed_results(all_results)

    # Save results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()

"""
Gated Attention Experiment: Compare standard vs gated attention mechanisms.

This script runs experiments on three synthetic tasks:
1. Copy Task - Tests basic information flow
2. Associative Recall - Tests key-value memory retrieval
3. Selective Copy - Tests distractor filtering (gating showcase)

Designed to run within a few hours on A100.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from experiments.synthetic_tasks import (
    CopyTaskDataset, AssociativeRecallDataset, SelectiveCopyDataset,
    collate_synthetic, PAD_IDX
)
from experiments.gated_transformer import GatedTransformer
from modelling.transformer import Transformer


@dataclass
class ModelConfig:
    """Model configuration (same for both standard and gated)."""
    vocab_size: int = 36
    d_model: int = 128
    n_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.0  # No dropout for fair comparison
    max_len: int = 256


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    eval_every: int = 500
    log_every: int = 100


@dataclass
class TaskConfig:
    """Task-specific configuration."""
    name: str
    dataset_class: type
    dataset_kwargs: dict
    num_train_samples: int
    num_val_samples: int
    training_steps: int


def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then linear decay scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    distractor_tokens: Optional[set] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: (batch, seq_len) predicted token ids
        targets: (batch, seq_len) target token ids
        loss_mask: (batch, seq_len) boolean mask for valid positions
        distractor_tokens: Optional set of distractor token IDs for leakage computation

    Returns:
        Dictionary with metrics
    """
    # Token accuracy
    correct = ((predictions == targets) & loss_mask).sum().item()
    total = loss_mask.sum().item()
    token_accuracy = correct / total if total > 0 else 0.0

    # Exact match accuracy (all tokens in sequence correct)
    seq_correct = ((predictions == targets) | ~loss_mask).all(dim=1)
    exact_match = seq_correct.float().mean().item()

    metrics = {
        'token_accuracy': token_accuracy,
        'exact_match': exact_match
    }

    # Leakage rate (for selective copy)
    if distractor_tokens is not None:
        in_distractor = torch.zeros_like(predictions, dtype=torch.bool)
        for tok in distractor_tokens:
            in_distractor |= (predictions == tok)
        leakage = (in_distractor & loss_mask).sum().item()
        metrics['leakage_rate'] = leakage / total if total > 0 else 0.0

    return metrics


class MetricsTracker:
    """Track metrics during training."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.token_accuracies = []
        self.exact_matches = []
        self.leakage_rates = []
        self.gate_means = []
        self.gate_stds = []
        self.steps = []

    def log(self, step: int, train_loss: float, val_loss: float, metrics: Dict,
            gate_stats: Optional[Dict] = None):
        self.steps.append(step)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.token_accuracies.append(metrics.get('token_accuracy', 0))
        self.exact_matches.append(metrics.get('exact_match', 0))
        self.leakage_rates.append(metrics.get('leakage_rate', None))

        if gate_stats:
            self.gate_means.append(gate_stats.get('gate_mean', 0.5))
            self.gate_stds.append(gate_stats.get('gate_std', 0.0))

    def to_dict(self) -> Dict:
        return {
            'steps': self.steps,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'token_accuracies': self.token_accuracies,
            'exact_matches': self.exact_matches,
            'leakage_rates': self.leakage_rates,
            'gate_means': self.gate_means,
            'gate_stds': self.gate_stds
        }


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_grad_norm: float,
    collect_gate_stats: bool = False
) -> tuple:
    """Execute one training step."""
    model.train()

    src = batch['src']
    tgt = batch['tgt']
    src_mask = batch['src_mask']
    tgt_mask = batch['tgt_mask']
    loss_mask = batch['loss_mask']

    # Teacher forcing: shift target
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    tgt_mask_input = tgt_mask[:, :-1]
    loss_mask_shifted = loss_mask[:, 1:]

    optimizer.zero_grad()

    # Forward pass
    gate_stats = None
    if collect_gate_stats and hasattr(model, 'forward_with_gate_stats'):
        logits, gate_stats = model.forward_with_gate_stats(src, tgt_input, src_mask, tgt_mask_input)
    else:
        logits = model(src, tgt_input, src_mask, tgt_mask_input)

    # Compute masked loss
    loss_per_token = criterion(logits.transpose(1, 2), tgt_output)  # (batch, seq_len)
    masked_loss = (loss_per_token * loss_mask_shifted.float()).sum() / loss_mask_shifted.sum()

    # Backward pass
    masked_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return masked_loss.item(), gate_stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    distractor_tokens: Optional[set] = None
) -> tuple:
    """Evaluate model on validation set."""
    model.eval()

    total_loss = 0
    total_tokens = 0
    all_predictions = []
    all_targets = []
    all_masks = []

    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask_input = tgt_mask[:, :-1]
        loss_mask_shifted = loss_mask[:, 1:]

        logits = model(src, tgt_input, src_mask, tgt_mask_input)

        # Compute loss
        loss_per_token = criterion(logits.transpose(1, 2), tgt_output)
        masked_loss = (loss_per_token * loss_mask_shifted.float()).sum()
        total_loss += masked_loss.item()
        total_tokens += loss_mask_shifted.sum().item()

        # Collect predictions for metrics
        predictions = logits.argmax(dim=-1)
        all_predictions.append(predictions)
        all_targets.append(tgt_output)
        all_masks.append(loss_mask_shifted)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    metrics = compute_metrics(all_predictions, all_targets, all_masks, distractor_tokens)

    return avg_loss, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: TrainingConfig,
    total_steps: int,
    device: torch.device,
    model_name: str = "Model",
    collect_gate_stats: bool = False,
    distractor_tokens: Optional[set] = None
) -> MetricsTracker:
    """Train model and track metrics."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    scheduler = get_linear_warmup_scheduler(
        optimizer, training_config.warmup_steps, total_steps
    )
    criterion = nn.CrossEntropyLoss(reduction='none')

    tracker = MetricsTracker()
    train_iter = iter(train_loader)
    running_loss = 0
    running_count = 0

    for step in range(1, total_steps + 1):
        # Get next batch (cycle through dataset)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Train step
        loss, gate_stats = train_step(
            model, batch, optimizer, criterion,
            training_config.max_grad_norm, collect_gate_stats
        )
        scheduler.step()

        running_loss += loss
        running_count += 1

        # Log progress
        if step % training_config.log_every == 0:
            avg_train_loss = running_loss / running_count
            gate_info = ""
            if gate_stats:
                gate_info = f", Gate: {gate_stats['gate_mean']:.3f}"
            print(f"  [{model_name}] Step {step}/{total_steps} - Loss: {avg_train_loss:.4f}{gate_info}")

        # Evaluate
        if step % training_config.eval_every == 0 or step == total_steps:
            avg_train_loss = running_loss / running_count
            val_loss, metrics = evaluate(model, val_loader, criterion, device, distractor_tokens)

            tracker.log(step, avg_train_loss, val_loss, metrics, gate_stats)

            leakage_info = ""
            if metrics.get('leakage_rate') is not None:
                leakage_info = f", Leak: {metrics['leakage_rate']:.4f}"

            print(f"  [{model_name}] Step {step} - Val Loss: {val_loss:.4f}, "
                  f"Acc: {metrics['token_accuracy']:.4f}, EM: {metrics['exact_match']:.4f}{leakage_info}")

            running_loss = 0
            running_count = 0

    return tracker


def run_task_experiment(
    task_config: TaskConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    device: torch.device
) -> Dict[str, Any]:
    """Run experiment for a single task comparing both models."""
    print(f"\n{'='*60}")
    print(f"Task: {task_config.name}")
    print(f"Training steps: {task_config.training_steps}")
    print(f"{'='*60}")

    # Create datasets with fixed seeds for reproducibility
    train_dataset = task_config.dataset_class(
        num_samples=task_config.num_train_samples,
        seed=42,
        **task_config.dataset_kwargs
    )
    val_dataset = task_config.dataset_class(
        num_samples=task_config.num_val_samples,
        seed=43,
        **task_config.dataset_kwargs
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_synthetic(b, PAD_IDX),
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_synthetic(b, PAD_IDX),
        num_workers=0
    )

    # Get distractor tokens for selective copy
    distractor_tokens = None
    if hasattr(train_dataset, 'get_distractor_token_set'):
        distractor_tokens = train_dataset.get_distractor_token_set()

    results = {'task': task_config.name}

    # Train standard model
    print("\n--- Standard Attention ---")
    standard_model = Transformer(**asdict(model_config)).to(device)
    results['standard'] = train_model(
        standard_model, train_loader, val_loader, training_config,
        task_config.training_steps, device, "Standard",
        collect_gate_stats=False, distractor_tokens=distractor_tokens
    ).to_dict()

    # Train gated model
    print("\n--- Gated Attention ---")
    gated_model = GatedTransformer(**asdict(model_config)).to(device)
    results['gated'] = train_model(
        gated_model, train_loader, val_loader, training_config,
        task_config.training_steps, device, "Gated",
        collect_gate_stats=True, distractor_tokens=distractor_tokens
    ).to_dict()

    return results


def generate_figures(all_results: Dict[str, Any], output_dir: str):
    """Generate comparison figures for all tasks."""
    os.makedirs(output_dir, exist_ok=True)

    for task_name, task_results in all_results.items():
        if task_name == 'metadata':
            continue

        std = task_results['standard']
        gated = task_results['gated']
        steps = std['steps']

        # Figure 1: Loss curves
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, std['train_losses'], 'b-', label='Standard Train', alpha=0.7)
        ax.plot(steps, std['val_losses'], 'b--', label='Standard Val', linewidth=2)
        ax.plot(steps, gated['train_losses'], 'r-', label='Gated Train', alpha=0.7)
        ax.plot(steps, gated['val_losses'], 'r--', label='Gated Val', linewidth=2)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title(f'{task_name}: Loss Curves')
        ax.legend()
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task_name}_loss.png'), dpi=150)
        plt.close()

        # Figure 2: Accuracy curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(steps, std['token_accuracies'], 'b-', label='Standard', linewidth=2)
        ax1.plot(steps, gated['token_accuracies'], 'r-', label='Gated', linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Token Accuracy')
        ax1.set_title(f'{task_name}: Token Accuracy')
        ax1.legend()
        ax1.set_ylim([0, 1.05])

        ax2.plot(steps, std['exact_matches'], 'b-', label='Standard', linewidth=2)
        ax2.plot(steps, gated['exact_matches'], 'r-', label='Gated', linewidth=2)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Exact Match Accuracy')
        ax2.set_title(f'{task_name}: Exact Match')
        ax2.legend()
        ax2.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task_name}_accuracy.png'), dpi=150)
        plt.close()

        # Figure 3: Gate statistics (if available)
        if gated['gate_means']:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, gated['gate_means'], 'g-', label='Gate Mean', linewidth=2)
            ax.fill_between(
                steps,
                [m - s for m, s in zip(gated['gate_means'], gated['gate_stds'])],
                [m + s for m, s in zip(gated['gate_means'], gated['gate_stds'])],
                alpha=0.3, color='g'
            )
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral (0.5)')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Gate Activation')
            ax.set_title(f'{task_name}: Gate Statistics')
            ax.legend()
            ax.set_ylim([0, 1])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{task_name}_gate.png'), dpi=150)
            plt.close()

        # Figure 4: Leakage rate (for selective copy)
        if std['leakage_rates'][0] is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, std['leakage_rates'], 'b-', label='Standard', linewidth=2)
            ax.plot(steps, gated['leakage_rates'], 'r-', label='Gated', linewidth=2)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Leakage Rate')
            ax.set_title(f'{task_name}: Distractor Leakage Rate')
            ax.legend()
            ax.set_ylim([0, max(max(std['leakage_rates']), max(gated['leakage_rates'])) * 1.1 + 0.01])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{task_name}_leakage.png'), dpi=150)
            plt.close()

    # Combined summary figure
    task_names = [k for k in all_results.keys() if k != 'metadata']
    if task_names:
        fig, axes = plt.subplots(1, len(task_names), figsize=(5*len(task_names), 5))
        if len(task_names) == 1:
            axes = [axes]

        for i, task_name in enumerate(task_names):
            ax = axes[i]
            std = all_results[task_name]['standard']
            gated = all_results[task_name]['gated']

            # Bar chart of final metrics
            metrics = ['Token Acc', 'Exact Match']
            x = range(len(metrics))
            std_vals = [std['token_accuracies'][-1], std['exact_matches'][-1]]
            gated_vals = [gated['token_accuracies'][-1], gated['exact_matches'][-1]]

            width = 0.35
            ax.bar([xi - width/2 for xi in x], std_vals, width, label='Standard', color='blue', alpha=0.7)
            ax.bar([xi + width/2 for xi in x], gated_vals, width, label='Gated', color='red', alpha=0.7)

            ax.set_ylabel('Score')
            ax.set_title(task_name)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=150)
        plt.close()

    print(f"\nFigures saved to: {output_dir}")


def print_summary(all_results: Dict[str, Any]):
    """Print summary table of results."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Task':<20} {'Model':<10} {'Token Acc':<12} {'Exact Match':<12} {'Leakage':<10}")
    print("-"*80)

    for task_name, task_results in all_results.items():
        if task_name == 'metadata':
            continue

        for model_name in ['standard', 'gated']:
            results = task_results[model_name]
            token_acc = results['token_accuracies'][-1]
            exact_match = results['exact_matches'][-1]
            leakage = results['leakage_rates'][-1]
            leakage_str = f"{leakage:.4f}" if leakage is not None else "N/A"

            print(f"{task_name:<20} {model_name:<10} {token_acc:<12.4f} {exact_match:<12.4f} {leakage_str:<10}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Compare standard vs gated attention')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output-dir', type=str, default='experiments/results',
                        help='Output directory for results')
    parser.add_argument('--tasks', nargs='+', default=['copy', 'associative', 'selective'],
                        choices=['copy', 'associative', 'selective'],
                        help='Tasks to run')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with reduced steps for testing')
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Model config
    model_config = ModelConfig()

    # Training config
    training_config = TrainingConfig()

    # Task configs
    step_multiplier = 0.1 if args.quick else 1.0
    sample_multiplier = 0.1 if args.quick else 1.0

    task_configs = {
        'copy': TaskConfig(
            name='copy_task',
            dataset_class=CopyTaskDataset,
            dataset_kwargs={'min_length': 16, 'max_length': 64, 'vocab_size': model_config.vocab_size},
            num_train_samples=int(50000 * sample_multiplier),
            num_val_samples=int(5000 * sample_multiplier),
            training_steps=int(10000 * step_multiplier)
        ),
        'associative': TaskConfig(
            name='associative_recall',
            dataset_class=AssociativeRecallDataset,
            dataset_kwargs={'min_pairs': 8, 'max_pairs': 16, 'vocab_size': model_config.vocab_size},
            num_train_samples=int(100000 * sample_multiplier),
            num_val_samples=int(10000 * sample_multiplier),
            training_steps=int(20000 * step_multiplier)
        ),
        'selective': TaskConfig(
            name='selective_copy',
            dataset_class=SelectiveCopyDataset,
            dataset_kwargs={'min_length': 32, 'max_length': 128,
                           'relevant_fraction': 0.5, 'vocab_size': model_config.vocab_size},
            num_train_samples=int(100000 * sample_multiplier),
            num_val_samples=int(10000 * sample_multiplier),
            training_steps=int(20000 * step_multiplier)
        )
    }

    # Run experiments
    all_results = {}
    for task_key in args.tasks:
        if task_key in task_configs:
            results = run_task_experiment(
                task_configs[task_key],
                model_config,
                training_config,
                device
            )
            all_results[task_configs[task_key].name] = results

    # Add metadata
    all_results['metadata'] = {
        'model_config': asdict(model_config),
        'training_config': asdict(training_config),
        'timestamp': datetime.now().isoformat(),
        'device': str(device)
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f'results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate figures
    figures_dir = os.path.join(args.output_dir, 'figures')
    generate_figures(all_results, figures_dir)

    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    main()

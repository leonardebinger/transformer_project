import torch
from torch.optim import Optimizer


class TransformerLRScheduler:
    """
    Learning rate scheduler from 'Attention Is All You Need'.

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0
        self._lr = 0

    def step(self):
        """Update the learning rate."""
        self._step += 1
        self._lr = self._compute_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._lr

    def _compute_lr(self) -> float:
        """Compute learning rate for current step."""
        step = self._step
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

    def get_lr(self) -> float:
        """Return the current learning rate."""
        return self._lr


def get_optimizer(model: torch.nn.Module, lr: float = 1e-4, weight_decay: float = 0.01):
    """
    Create AdamW optimizer with no weight decay on bias and LayerNorm parameters.

    Args:
        model: The transformer model
        lr: Learning rate (will be overridden by scheduler)
        weight_decay: Weight decay for non-bias, non-LayerNorm parameters

    Returns:
        AdamW optimizer
    """
    # Separate parameters into two groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for bias and LayerNorm parameters
        if 'bias' in name or 'layer_norm' in name or 'LayerNorm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-9)

    return optimizer

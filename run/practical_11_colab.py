"""
Practical 11: GPU Training and Mixed Precision Training
========================================================

This script is designed to run in Google Colab with a T4 GPU.
It demonstrates:
1. GPU training
2. Mixed precision training with torch.cuda.amp
"""

import torch
from transformers import (RobertaForSequenceClassification, RobertaTokenizer,
                          get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader
from datasets import load_dataset
import tqdm
from sklearn.metrics import f1_score

# Configuration
MODEL_NAME_OR_PATH = 'roberta-base'
MAX_INPUT_LENGTH = 256
BATCH_SIZE = 16
TRAINING_EPOCHS = 2
WEIGHT_DECAY = 0.01
LEARNING_RATE = 2e-5
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0

# Exercise 2: Set device to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Exercise 3: Enable mixed precision only on GPU
MIXED_PRECISION_TRAINING = torch.cuda.is_available()
print(f"Mixed precision training: {MIXED_PRECISION_TRAINING}")

# Initialize model and tokenizer
print("Loading model and tokenizer...")
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

# Exercise 2: Move model to GPU
model = model.to(DEVICE)

# Load dataset
print("Loading QNLI dataset...")
qnli_dataset = load_dataset('glue', 'qnli')


def convert_example_to_features(example: dict) -> dict:
    """Convert example to features."""
    features = tokenizer(
        example['question'],
        example['sentence'],
        max_length=MAX_INPUT_LENGTH,
        padding='max_length',
        truncation='longest_first'
    )
    features['labels'] = example['label']
    return features


def collate(batch: list) -> dict:
    """Collate batch and move to device."""
    features = {
        'input_ids': torch.tensor([itm['input_ids'] for itm in batch]),
        'attention_mask': torch.tensor([itm['attention_mask'] for itm in batch]),
        'labels': torch.tensor([itm['labels'] for itm in batch]),
    }
    return features


# Apply tokenization to the datasets
print("Tokenizing datasets...")
train_dataset = qnli_dataset['train'].map(convert_example_to_features)
validation_dataset = qnli_dataset['validation'].map(convert_example_to_features)

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, collate_fn=collate
)

# Setup optimizer with weight decay
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
        "lr": LEARNING_RATE
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": LEARNING_RATE
    },
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

# Learning rate scheduler
num_training_steps = len(train_dataloader) * TRAINING_EPOCHS
num_warmup_steps = int(WARMUP_PROPORTION * num_training_steps)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Exercise 3: Initialize GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION_TRAINING)


def training_step(batch):
    """
    Perform a training step with GPU and mixed precision support.

    Args:
        batch (dict): Batch of data.
    Returns:
        loss (torch.Tensor): Loss for the batch.
    """
    # Exercise 2: Move batch to GPU
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    # Exercise 3: Use autocast for mixed precision forward pass
    with torch.cuda.amp.autocast(enabled=MIXED_PRECISION_TRAINING):
        loss = model(**batch).loss

    # Exercise 3: Scale loss and backward pass for mixed precision
    scaler.scale(loss).backward()

    # Clip gradients (unscale first for proper clipping)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

    # Exercise 3: Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()

    lr_scheduler.step()
    model.zero_grad()

    return loss


def evaluate(dataloader):
    """
    Evaluate the model with GPU support.

    Args:
        dataloader: Dataloader for the data.
    Returns:
        f1 (float): F1 Score for the model.
    """
    model.eval()
    predictions = []
    labels = []

    for batch in tqdm.tqdm(dataloader, desc="Eval"):
        # Exercise 2: Move batch to GPU
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.no_grad():
            # Exercise 3: Use autocast for mixed precision inference
            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION_TRAINING):
                logits = model(**batch).logits
            logits = logits.detach().cpu()

        pred = logits.argmax(-1)
        predictions.append(pred.reshape(-1))
        labels.append(batch['labels'].cpu().reshape(-1))

    model.zero_grad()
    model.train()

    predictions = torch.cat(predictions, 0)
    labels = torch.cat(labels, 0)
    f1 = f1_score(labels.numpy(), predictions.numpy())

    return f1


def main():
    """Main training loop."""
    print(f"\nStarting training on {DEVICE}...")
    print(f"Mixed precision: {MIXED_PRECISION_TRAINING}")
    print(f"Training epochs: {TRAINING_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total training steps: {num_training_steps}")

    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    for epoch in range(TRAINING_EPOCHS):
        iterator = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{TRAINING_EPOCHS}")

        for batch in iterator:
            loss = training_step(batch)
            iterator.set_postfix({'Loss': loss.item()})

        # Evaluate after each epoch
        f1 = evaluate(validation_dataloader)
        print(f"Epoch {epoch+1} - Validation F1 Score: {f1:.4f}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()

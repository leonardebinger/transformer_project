# Transformer Implementation Project

Implementation of a transformer model from scratch for German-English translation, following the "Attention Is All You Need" paper.

## Project Structure

```
├── modelling/
│   ├── attention.py          # Scaled dot-product & multi-head attention
│   ├── gated_attention.py    # Gated attention mechanisms (experiment)
│   ├── positional_encoding.py # Sinusoidal positional encoding
│   ├── embedding.py          # Token embedding layer
│   ├── functional.py         # Encoder & decoder layers
│   ├── transformer.py        # Full transformer model
│   ├── scheduler.py          # LR scheduler & optimizer setup
│   ├── tokenizer.py          # BPE tokenizer (custom + HuggingFace GPT2)
│   └── generation.py         # Greedy decoding & BLEU evaluation
├── dataset.py                # Data cleaning, vocabulary, dataset class
├── run/
│   ├── main.py               # Training with synthetic data
│   ├── train_wmt.py          # WMT17 DE-EN training with results saving
│   ├── generate_figures.py   # Generate figures for report
│   └── practical_11_colab.py # GPU & mixed precision training (Colab)
├── experiments/
│   ├── synthetic_tasks.py    # Copy task & associative recall datasets
│   ├── gated_transformer.py  # Gated attention transformer model
│   └── compare_attention.py  # Standard vs gated attention experiments
├── results/                  # Training results (JSON) and saved models
├── figures/                  # Generated figures for report
└── test/                     # Provided test files
```

## Usage

```bash
# Run tests
python -m pytest test/ -v

# Train on synthetic data (quick test)
python run/main.py

# Train on WMT17 German-English (full training)
python run/train_wmt.py

# Run gated attention experiments
python experiments/compare_attention.py

# Generate figures for report
python run/generate_figures.py
```

## Gated Attention Experiment

The `experiments/` directory contains code for comparing standard vs gated attention:
- **Copy Task**: Model copies input to output
- **Associative Recall**: Model recalls value associated with a queried key

Results show gated attention achieves 94% accuracy on copy task vs 70% for standard attention.

## Notes

- AI coding tools were used for understanding, writing, reviewing, and refactoring code
- Test tolerance in `test/practical_6.py` was adjusted (`atol=1e-4`) due to floating point precision differences

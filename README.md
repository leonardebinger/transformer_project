# Transformer Implementation Project

Implementation of a transformer model from scratch for German-English translation, following the "Attention Is All You Need" paper.

## Project Structure

```
├── modelling/
│   ├── attention.py          # Scaled dot-product & multi-head attention
│   ├── positional_encoding.py # Sinusoidal positional encoding
│   ├── embedding.py          # Token embedding layer
│   ├── functional.py         # Encoder & decoder layers
│   ├── transformer.py        # Full transformer model
│   ├── scheduler.py          # LR scheduler & optimizer setup
│   ├── tokenizer.py          # BPE tokenizer implementation
│   └── generation.py         # Greedy decoding & BLEU evaluation
├── dataset.py                # Data cleaning, vocabulary, dataset class
├── run/
│   ├── main.py               # Training & evaluation script
│   └── practical_11_colab.py # GPU & mixed precision training (Colab)
└── test/                     # Provided test files
```

## Usage

```bash
# Run tests
python -m pytest test/ -v

# Train model (synthetic data)
python run/main.py
```

## Notes

- AI coding tools were used for understanding, writing, reviewing, and refactoring code
- Test tolerance in `test/practical_6.py` was adjusted (`atol=1e-4`) due to floating point precision differences

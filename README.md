# Transformer Implementation Project

Implementation of a transformer model from scratch for German-English translation, following the "Attention Is All You Need" paper, with gated attention experiments extension.

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
│   └── practical_11_colab.py # GPU & mixed precision training (Colab)
├── experiments/
│   ├── synthetic_tasks.py    # Copy task & associative recall datasets
│   ├── gated_transformer.py  # Gated attention transformer model
│   └── compare_attention.py  # Standard vs gated attention experiments
├── test/
│   ├── practical_2_test.py
│   ├── practical_4_test.py
│   ├── practical_5_encoder_layer_tests.py
│   ├── practical_5_mha_tests.py
│   └── practical_6.py
└── requirements.txt
```

## Notes

- AI coding tools were used for understanding, writing, reviewing, and refactoring code
- Test tolerance in `test/practical_6.py` was adjusted (`atol=1e-4`) due to floating point precision differences

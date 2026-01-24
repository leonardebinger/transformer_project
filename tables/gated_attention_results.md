# Gated Attention Experiment Results

## Summary Table

| Task | Model | Final Accuracy | Final Val Loss | Epochs to 90% | Loss Decrease/Epoch |
|------|-------|----------------|----------------|---------------|---------------------|
| Copy Task | Standard | 69.6% | 1.666 | 19 | 0.1022 |
| Copy Task | Gated | **94.2%** | 0.664 | 19 | 0.0840 |
| Associative Recall | Standard | 52.8% | 2.327 | 1 | 0.2293 |
| Associative Recall | Gated | 53.2% | 2.365 | 1 | 0.2049 |

## Key Findings

- **Copy Task**: Gated attention achieves significantly higher accuracy (94.2% vs 69.6%)
- **Associative Recall**: Both models perform similarly (~52-53%), suggesting task complexity
- Gate activations stabilize around 0.53 for copy task, indicating learned selective attention

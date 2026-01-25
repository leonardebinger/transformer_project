# Gated Attention Experiment Results

## Summary Table

| Task | Model | Final Accuracy | Final Val Loss | Epochs to 90% | Loss Decrease/Epoch |
|------|-------|----------------|----------------|---------------|---------------------|
| Copy Task | Standard | 73.3% | 1.529 | 19 | 0.0865 |
| Copy Task | Gated | **93.1%** | 0.773 | 18 | 0.0806 |
| Associative Recall | Standard | 60.0% | 1.813 | 6 | 0.2214 |
| Associative Recall | Gated | 65.2% | 1.609 | 14 | 0.2503 |

## Key Findings

- **Copy Task**: Gated attention achieves significantly higher accuracy (+19.8%)
- **Associative Recall**: Gated attention achieves significantly higher accuracy (+5.2%)

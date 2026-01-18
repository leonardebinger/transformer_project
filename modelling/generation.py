import torch
from typing import List


def greedy_decode(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
    max_len: int = 64,
    device: torch.device = None
) -> torch.Tensor:
    """
    Greedy autoregressive decoding.

    Args:
        model: Transformer model
        src: Source token ids (batch, src_len)
        src_mask: Source padding mask (batch, src_len)
        bos_idx: Beginning of sequence token index
        eos_idx: End of sequence token index
        max_len: Maximum generation length
        device: Device to use

    Returns:
        Generated token ids (batch, generated_len)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    batch_size = src.size(0)

    # Encode source
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)

    # Initialize decoder input with BOS token
    tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

    # Track which sequences have finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        # Create target mask (all ones since we're generating)
        tgt_mask = torch.ones(batch_size, tgt.size(1), device=device)

        # Decode
        with torch.no_grad():
            decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            logits = model.output_projection(decoder_output)

        # Get next token (greedy: take argmax of last position)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Append to sequence
        tgt = torch.cat([tgt, next_token], dim=1)

        # Check for EOS
        finished = finished | (next_token.squeeze(-1) == eos_idx)
        if finished.all():
            break

    return tgt


def translate_sentence(
    model,
    sentence: str,
    vocab,
    max_len: int = 64,
    device: torch.device = None
) -> str:
    """
    Translate a single sentence.

    Args:
        model: Transformer model
        sentence: Source sentence string
        vocab: Vocabulary object
        max_len: Maximum generation length
        device: Device to use

    Returns:
        Translated sentence string
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode source sentence
    src_ids = vocab.encode(sentence)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = torch.ones(1, len(src_ids), device=device)

    # Generate translation
    output_ids = greedy_decode(
        model, src, src_mask,
        bos_idx=vocab.bos_idx,
        eos_idx=vocab.eos_idx,
        max_len=max_len,
        device=device
    )

    # Decode output (skip BOS, stop at EOS)
    output_ids = output_ids[0].tolist()
    if vocab.bos_idx in output_ids:
        output_ids = output_ids[output_ids.index(vocab.bos_idx) + 1:]
    if vocab.eos_idx in output_ids:
        output_ids = output_ids[:output_ids.index(vocab.eos_idx)]

    return vocab.decode(output_ids)


def translate_batch(
    model,
    sentences: List[str],
    vocab,
    max_len: int = 64,
    device: torch.device = None
) -> List[str]:
    """
    Translate a batch of sentences.

    Args:
        model: Transformer model
        sentences: List of source sentences
        vocab: Vocabulary object
        max_len: Maximum generation length
        device: Device to use

    Returns:
        List of translated sentences
    """
    translations = []
    for sentence in sentences:
        translation = translate_sentence(model, sentence, vocab, max_len, device)
        translations.append(translation)
    return translations


def compute_bleu(predictions: List[str], references: List[str]) -> dict:
    """
    Compute BLEU score using HuggingFace evaluate.

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        Dictionary with BLEU score and related metrics
    """
    import evaluate

    # Handle empty predictions by replacing with placeholder
    predictions_clean = []
    for pred in predictions:
        if not pred or not pred.strip():
            predictions_clean.append("<empty>")
        else:
            predictions_clean.append(pred)

    bleu = evaluate.load("bleu")

    # BLEU expects references as list of lists (multiple references per prediction)
    references_formatted = [[ref] for ref in references]

    try:
        results = bleu.compute(predictions=predictions_clean, references=references_formatted)
    except ZeroDivisionError:
        # Handle case where predictions are too short
        results = {
            'bleu': 0.0,
            'precisions': [0.0, 0.0, 0.0, 0.0],
            'brevity_penalty': 0.0,
            'length_ratio': 0.0,
            'translation_length': 0,
            'reference_length': sum(len(ref.split()) for ref in references)
        }

    return results

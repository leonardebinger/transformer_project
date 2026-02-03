import re
from typing import List, Tuple, Dict, Optional
from collections import Counter

import torch
from torch.utils.data import Dataset


# Character whitelist from practical specification
WHITELIST = "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥'\""

# Special tokens
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"

# Length constraints
MIN_LENGTH = 5
MAX_LENGTH = 64


def clean_text(text: str) -> str:
    """Clean a single text string."""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Keep only whitelisted characters
    text = ''.join(c for c in text if c in WHITELIST)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def is_valid_pair(src: str, tgt: str, min_len: int = MIN_LENGTH,
                  max_len: int = MAX_LENGTH, max_ratio: float = 3.0) -> bool:
    """Check if a sentence pair is valid based on length and ratio criteria."""
    src_tokens = src.split()
    tgt_tokens = tgt.split()

    src_len = len(src_tokens)
    tgt_len = len(tgt_tokens)

    # Check length constraints
    if src_len < min_len or src_len > max_len:
        return False
    if tgt_len < min_len or tgt_len > max_len:
        return False

    # Check ratio
    if src_len == 0 or tgt_len == 0:
        return False
    ratio = max(src_len / tgt_len, tgt_len / src_len)
    if ratio > max_ratio:
        return False

    return True


def clean_dataset(data: List[Dict]) -> List[Tuple[str, str]]:
    """
    Clean a dataset of translation pairs.

    Args:
        data: List of dicts with 'translation' key containing 'de' and 'en' keys

    Returns:
        List of (source, target) tuples that pass validation
    """
    cleaned = []
    for item in data:
        if 'translation' in item:
            src = clean_text(item['translation']['de'])
            tgt = clean_text(item['translation']['en'])
        else:
            src = clean_text(item.get('de', ''))
            tgt = clean_text(item.get('en', ''))

        if is_valid_pair(src, tgt):
            cleaned.append((src, tgt))

    return cleaned


class Vocabulary:
    """Vocabulary class for mapping tokens to indices."""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.token2idx = {}
        self.idx2token = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize special tokens."""
        special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        for idx, token in enumerate(special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build(self, texts: List[str]):
        """Build vocabulary from texts."""
        # Count all tokens
        counter = Counter()
        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)

        # Get top vocab_size - num_special_tokens most common
        num_special = len(self.token2idx)
        most_common = counter.most_common(self.vocab_size - num_special)

        # Add to vocabulary
        for token, _ in most_common:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def encode(self, text: str) -> List[int]:
        """Encode text to token indices."""
        tokens = text.lower().split()
        unk_idx = self.token2idx[UNK_TOKEN]
        return [self.token2idx.get(t, unk_idx) for t in tokens]

    def decode(self, indices: List[int]) -> str:
        """Decode token indices to text."""
        tokens = [self.idx2token.get(idx, UNK_TOKEN) for idx in indices]
        return ' '.join(tokens)

    @property
    def pad_idx(self) -> int:
        return self.token2idx[PAD_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.token2idx[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token2idx)


class TranslationDataset(Dataset):
    """PyTorch Dataset for translation task."""

    def __init__(self, data: List[Tuple[str, str]], vocab: Vocabulary,
                 max_len: int = MAX_LENGTH):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src, tgt = self.data[idx]

        # Encode source and target
        src_ids = self.vocab.encode(src)
        tgt_ids = self.vocab.encode(tgt)

        # Add BOS and EOS to target
        tgt_ids = [self.vocab.bos_idx] + tgt_ids + [self.vocab.eos_idx]

        # Truncate if necessary
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]

        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'src_len': len(src_ids),
            'tgt_len': len(tgt_ids)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader to handle padding."""
    # Find max lengths in batch
    max_src_len = max(item['src_len'] for item in batch)
    max_tgt_len = max(item['tgt_len'] for item in batch)

    # Pad sequences
    src_padded = []
    tgt_padded = []
    src_mask = []
    tgt_mask = []

    pad_idx = 0  # PAD token index

    for item in batch:
        # Pad source
        src = item['src']
        src_pad_len = max_src_len - len(src)
        src_padded.append(torch.cat([src, torch.full((src_pad_len,), pad_idx, dtype=torch.long)]))
        src_mask.append(torch.cat([torch.ones(len(src)), torch.zeros(src_pad_len)]))

        # Pad target
        tgt = item['tgt']
        tgt_pad_len = max_tgt_len - len(tgt)
        tgt_padded.append(torch.cat([tgt, torch.full((tgt_pad_len,), pad_idx, dtype=torch.long)]))
        tgt_mask.append(torch.cat([torch.ones(len(tgt)), torch.zeros(tgt_pad_len)]))

    return {
        'src': torch.stack(src_padded),
        'tgt': torch.stack(tgt_padded),
        'src_mask': torch.stack(src_mask),
        'tgt_mask': torch.stack(tgt_mask)
    }

"""
Synthetic tasks for evaluating attention mechanisms.

Three tasks designed to test different aspects of attention:
1. Copy Task - Tests basic information flow through attention
2. Associative Recall - Tests key-value memory retrieval
3. Selective Copy - Tests ability to filter distractors (gating showcase)
"""

import torch
from torch.utils.data import Dataset
import random
from typing import Dict, List, Optional

# Special token indices (fixed across all tasks)
PAD_IDX = 0      # Padding token
SEP_IDX = 1      # Separator / delimiter
BOS_IDX = 2      # Beginning of sequence (decoder start)
EOS_IDX = 3      # End of sequence
MARKER_IDX = 4   # Marker for relevant tokens (selective copy)

# Content tokens start at index 5
CONTENT_START = 5


class CopyTaskDataset(Dataset):
    """
    Copy task: Model must reproduce input sequence in output.

    Input:  [t1, t2, ..., tL, SEP, PAD...]
    Target: [BOS, t1, t2, ..., tL, EOS, PAD...]

    Loss is computed only on positions after BOS (the copied tokens + EOS).
    """

    def __init__(
        self,
        num_samples: int,
        min_length: int = 16,
        max_length: int = 64,
        vocab_size: int = 36,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Content token range: [CONTENT_START, vocab_size-1]
        self.content_tokens = list(range(CONTENT_START, vocab_size))

        if seed is not None:
            random.seed(seed)

        self.data = self._generate_data()

    def _generate_data(self) -> List[Dict]:
        data = []
        for _ in range(self.num_samples):
            # Random sequence length
            length = random.randint(self.min_length, self.max_length)

            # Generate random content tokens
            content = [random.choice(self.content_tokens) for _ in range(length)]

            # Source: [content..., SEP]
            src = content + [SEP_IDX]

            # Target: [BOS, content..., EOS]
            tgt = [BOS_IDX] + content + [EOS_IDX]

            # Loss mask: compute loss on all positions after BOS (i.e., content + EOS)
            # Target is [BOS, t1, t2, ..., tL, EOS]
            # We compute loss on positions 1 to L+1 (the content and EOS)
            loss_mask = [False] + [True] * (length + 1)

            data.append({
                'src': src,
                'tgt': tgt,
                'loss_mask': loss_mask
            })

        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            'src': torch.tensor(item['src'], dtype=torch.long),
            'tgt': torch.tensor(item['tgt'], dtype=torch.long),
            'loss_mask': torch.tensor(item['loss_mask'], dtype=torch.bool)
        }


class AssociativeRecallDataset(Dataset):
    """
    Associative recall: Given key-value pairs, query a key and return the NEXT value.

    Input:  [k1a, k1b, v1a, v1b, SEP, k2a, k2b, v2a, v2b, SEP, ..., SEP, qa, qb, SEP]
    Target: [BOS, v_{i+1}a, v_{i+1}b, EOS]

    Where querying key_i returns value_{(i+1) mod n} (the next value, wrapping around).
    This tests the model's ability to attend to the correct key and retrieve associated value.
    """

    def __init__(
        self,
        num_samples: int,
        min_pairs: int = 8,
        max_pairs: int = 32,
        key_length: int = 2,
        value_length: int = 2,
        vocab_size: int = 36,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.min_pairs = min_pairs
        self.max_pairs = max_pairs
        self.key_length = key_length
        self.value_length = value_length
        self.vocab_size = vocab_size

        # Content token range
        self.content_tokens = list(range(CONTENT_START, vocab_size))

        if seed is not None:
            random.seed(seed)

        self.data = self._generate_data()

    def _generate_data(self) -> List[Dict]:
        data = []
        for _ in range(self.num_samples):
            # Random number of pairs
            num_pairs = random.randint(self.min_pairs, self.max_pairs)

            # Generate unique keys and values (as tuples of tokens)
            # Need enough tokens for all keys and values to be unique
            all_tokens = self.content_tokens.copy()
            random.shuffle(all_tokens)

            keys = []
            values = []
            token_idx = 0

            for _ in range(num_pairs):
                # Each key is key_length tokens
                key = all_tokens[token_idx:token_idx + self.key_length]
                token_idx += self.key_length
                keys.append(key)

                # Each value is value_length tokens
                value = all_tokens[token_idx:token_idx + self.value_length]
                token_idx += self.value_length
                values.append(value)

            # Choose a random query key index
            query_idx = random.randint(0, num_pairs - 1)
            query_key = keys[query_idx]

            # Answer is the NEXT value (wrapping around)
            answer_idx = (query_idx + 1) % num_pairs
            answer_value = values[answer_idx]

            # Build source: [k1, v1, SEP, k2, v2, SEP, ..., SEP, query_key, SEP]
            src = []
            for k, v in zip(keys, values):
                src.extend(k)
                src.extend(v)
                src.append(SEP_IDX)
            src.extend(query_key)
            src.append(SEP_IDX)

            # Build target: [BOS, answer_value, EOS]
            tgt = [BOS_IDX] + answer_value + [EOS_IDX]

            # Loss mask: compute loss on answer value tokens and EOS
            loss_mask = [False] + [True] * (self.value_length + 1)

            data.append({
                'src': src,
                'tgt': tgt,
                'loss_mask': loss_mask
            })

        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            'src': torch.tensor(item['src'], dtype=torch.long),
            'tgt': torch.tensor(item['tgt'], dtype=torch.long),
            'loss_mask': torch.tensor(item['loss_mask'], dtype=torch.bool)
        }


class SelectiveCopyDataset(Dataset):
    """
    Selective copy: Filter relevant tokens from distractors.

    Input:  [MARKER, r1, d1, d2, MARKER, r2, d3, MARKER, r3, d4, d5, SEP, PAD...]
    Target: [BOS, r1, r2, r3, EOS, PAD...]

    MARKER token precedes each relevant token. Distractors appear without markers.
    Relevant and distractor tokens come from DISJOINT ranges.

    This task is where gating should visibly help - the model must learn to
    suppress distractor contributions and focus on marked relevant tokens.
    """

    def __init__(
        self,
        num_samples: int,
        min_length: int = 32,
        max_length: int = 128,
        relevant_fraction: float = 0.5,
        vocab_size: int = 36,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.relevant_fraction = relevant_fraction
        self.vocab_size = vocab_size

        # Use SHARED token range for both relevant and distractor tokens.
        # This forces the model to rely on the MARKER token for filtering
        # rather than learning token identity shortcuts.
        content_tokens = list(range(CONTENT_START, vocab_size))
        self.relevant_tokens = content_tokens
        self.distractor_tokens = content_tokens

        if seed is not None:
            random.seed(seed)

        self.data = self._generate_data()

    def _generate_data(self) -> List[Dict]:
        data = []
        for _ in range(self.num_samples):
            # Random total length (before adding markers)
            total_content = random.randint(self.min_length, self.max_length)

            # Number of relevant tokens
            num_relevant = max(1, int(total_content * self.relevant_fraction))
            num_distractors = total_content - num_relevant

            # Generate relevant and distractor tokens
            relevant_list = [random.choice(self.relevant_tokens) for _ in range(num_relevant)]
            distractor_list = [random.choice(self.distractor_tokens) for _ in range(num_distractors)]

            # Build source by interleaving: each relevant token is preceded by MARKER
            # Distractors are scattered between marked tokens
            src = []
            relevant_positions = []
            distractor_positions = []

            # Create a mixed sequence
            relevant_iter = iter(relevant_list)
            distractor_iter = iter(distractor_list)
            remaining_relevant = num_relevant
            remaining_distractors = num_distractors

            while remaining_relevant > 0 or remaining_distractors > 0:
                # Decide whether to add a relevant token (with marker) or distractor
                if remaining_relevant > 0 and remaining_distractors > 0:
                    # Add based on probability to roughly maintain fraction
                    add_relevant = random.random() < (remaining_relevant / (remaining_relevant + remaining_distractors))
                elif remaining_relevant > 0:
                    add_relevant = True
                else:
                    add_relevant = False

                if add_relevant:
                    # Add MARKER then relevant token
                    src.append(MARKER_IDX)
                    relevant_positions.append(len(src))  # Position of the relevant token
                    src.append(next(relevant_iter))
                    remaining_relevant -= 1
                else:
                    # Add distractor (no marker)
                    distractor_positions.append(len(src))
                    src.append(next(distractor_iter))
                    remaining_distractors -= 1

            # Add separator at end
            src.append(SEP_IDX)

            # Target: [BOS, relevant tokens in order, EOS]
            tgt = [BOS_IDX] + relevant_list + [EOS_IDX]

            # Loss mask: compute loss on relevant tokens and EOS
            loss_mask = [False] + [True] * (num_relevant + 1)

            data.append({
                'src': src,
                'tgt': tgt,
                'loss_mask': loss_mask,
                'relevant_positions': relevant_positions,
                'distractor_positions': distractor_positions,
                'num_relevant': num_relevant,
                'num_distractors': num_distractors
            })

        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            'src': torch.tensor(item['src'], dtype=torch.long),
            'tgt': torch.tensor(item['tgt'], dtype=torch.long),
            'loss_mask': torch.tensor(item['loss_mask'], dtype=torch.bool),
            'num_relevant': item['num_relevant'],
            'num_distractors': item['num_distractors']
        }

    def get_distractor_token_set(self) -> None:
        """No longer applicable — relevant and distractor tokens share the same vocabulary."""
        return None


def collate_synthetic(batch: List[Dict], pad_idx: int = PAD_IDX) -> Dict[str, torch.Tensor]:
    """
    Collate function for synthetic tasks with padding.

    Args:
        batch: List of sample dicts from dataset
        pad_idx: Padding token index

    Returns:
        Dictionary with padded tensors:
        - src: (batch, max_src_len)
        - tgt: (batch, max_tgt_len)
        - src_mask: (batch, max_src_len) - 1 for valid, 0 for pad
        - tgt_mask: (batch, max_tgt_len) - 1 for valid, 0 for pad
        - loss_mask: (batch, max_tgt_len) - True where loss should be computed
    """
    src_list = [item['src'] for item in batch]
    tgt_list = [item['tgt'] for item in batch]
    loss_mask_list = [item['loss_mask'] for item in batch]

    # Find max lengths
    max_src_len = max(len(s) for s in src_list)
    max_tgt_len = max(len(t) for t in tgt_list)

    batch_size = len(batch)

    # Pad sequences
    src_padded = torch.full((batch_size, max_src_len), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((batch_size, max_tgt_len), pad_idx, dtype=torch.long)
    src_mask = torch.zeros(batch_size, max_src_len, dtype=torch.long)
    tgt_mask = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)
    loss_mask = torch.zeros(batch_size, max_tgt_len, dtype=torch.bool)

    for i, (src, tgt, lm) in enumerate(zip(src_list, tgt_list, loss_mask_list)):
        src_len = len(src)
        tgt_len = len(tgt)

        src_padded[i, :src_len] = src
        tgt_padded[i, :tgt_len] = tgt
        src_mask[i, :src_len] = 1
        tgt_mask[i, :tgt_len] = 1
        loss_mask[i, :len(lm)] = lm

    result = {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'loss_mask': loss_mask
    }

    # Include selective copy specific fields if present
    if 'num_relevant' in batch[0]:
        result['num_relevant'] = torch.tensor([item['num_relevant'] for item in batch])
        result['num_distractors'] = torch.tensor([item['num_distractors'] for item in batch])

    return result

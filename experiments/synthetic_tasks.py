import torch
from torch.utils.data import Dataset
import random


class CopyTaskDataset(Dataset):
    """
    Copy task: Model must copy input sequence to output.

    Input:  [BOS] a b c d [SEP]
    Output: [BOS] a b c d [EOS]

    This tests whether the model can learn to pass information through attention.
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int,
                 pad_idx: int = 0, bos_idx: int = 1, eos_idx: int = 2, sep_idx: int = 3):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.sep_idx = sep_idx

        # Token range for actual content (excluding special tokens)
        self.min_token = 4
        self.max_token = vocab_size - 1

        # Pre-generate data
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Random sequence length (at least 3 tokens)
            length = random.randint(3, self.seq_len - 2)

            # Generate random content tokens
            content = [random.randint(self.min_token, self.max_token) for _ in range(length)]

            # Source: [BOS] content [SEP]
            src = [self.bos_idx] + content + [self.sep_idx]

            # Target: [BOS] content [EOS]
            tgt = [self.bos_idx] + content + [self.eos_idx]

            data.append((src, tgt))

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return {
            'src': torch.tensor(src, dtype=torch.long),
            'tgt': torch.tensor(tgt, dtype=torch.long)
        }


class AssociativeRecallDataset(Dataset):
    """
    Associative recall task: Given key-value pairs and a query key, recall the value.

    Input:  [BOS] k1 v1 k2 v2 k3 v3 [SEP] k2 [SEP]
    Output: [BOS] v2 [EOS]

    This tests the model's ability to store and retrieve associations via attention.
    """

    def __init__(self, num_samples: int, num_pairs: int, vocab_size: int,
                 pad_idx: int = 0, bos_idx: int = 1, eos_idx: int = 2, sep_idx: int = 3):
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.sep_idx = sep_idx

        self.min_token = 4
        self.max_token = vocab_size - 1

        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # Generate unique keys and values
            keys = random.sample(range(self.min_token, self.max_token), self.num_pairs)
            values = random.sample(range(self.min_token, self.max_token), self.num_pairs)

            # Choose a random query key
            query_idx = random.randint(0, self.num_pairs - 1)
            query_key = keys[query_idx]
            answer_value = values[query_idx]

            # Build source: [BOS] k1 v1 k2 v2 ... [SEP] query_key [SEP]
            src = [self.bos_idx]
            for k, v in zip(keys, values):
                src.extend([k, v])
            src.extend([self.sep_idx, query_key, self.sep_idx])

            # Build target: [BOS] answer_value [EOS]
            tgt = [self.bos_idx, answer_value, self.eos_idx]

            data.append((src, tgt))

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return {
            'src': torch.tensor(src, dtype=torch.long),
            'tgt': torch.tensor(tgt, dtype=torch.long)
        }


def collate_synthetic(batch, pad_idx=0):
    """Collate function for synthetic tasks with padding."""
    src_list = [item['src'] for item in batch]
    tgt_list = [item['tgt'] for item in batch]

    # Find max lengths
    max_src_len = max(len(s) for s in src_list)
    max_tgt_len = max(len(t) for t in tgt_list)

    # Pad sequences
    src_padded = torch.full((len(batch), max_src_len), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt_len), pad_idx, dtype=torch.long)
    src_mask = torch.zeros(len(batch), max_src_len, dtype=torch.long)
    tgt_mask = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_list, tgt_list)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
        src_mask[i, :len(src)] = 1
        tgt_mask[i, :len(tgt)] = 1

    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask
    }

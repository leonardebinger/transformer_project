from collections import Counter
from typing import List, Dict, Tuple


class BPETokenizer:
    """Byte Pair Encoding tokenizer implementation."""

    def __init__(self):
        self.vocab = []
        self.merges = {}  # (token1, token2) -> merged_token

    def _get_word_freqs(self, corpus: List[str]) -> Dict[str, int]:
        """Extract word frequencies from corpus."""
        word_freqs = Counter()
        for sentence in corpus:
            words = sentence.lower().split()
            word_freqs.update(words)
        return dict(word_freqs)

    def _get_pair_freqs(self, word_splits: Dict[str, List[str]],
                        word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Compute co-occurrence frequencies of adjacent token pairs."""
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            tokens = word_splits[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        return dict(pair_freqs)

    def _merge_pair(self, word_splits: Dict[str, List[str]],
                    pair: Tuple[str, str]) -> Dict[str, List[str]]:
        """Apply a merge rule to all word splits."""
        merged_token = pair[0] + pair[1]
        new_splits = {}
        for word, tokens in word_splits.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_splits[word] = new_tokens
        return new_splits

    def train(self, corpus: List[str], vocab_size: int):
        """Train the BPE tokenizer on a corpus."""
        # Get word frequencies
        word_freqs = self._get_word_freqs(corpus)

        # Build base vocabulary (unique characters)
        base_vocab = set()
        for word in word_freqs.keys():
            for char in word:
                base_vocab.add(char)
        self.vocab = sorted(list(base_vocab))

        # Initialize word splits (each word split into characters)
        word_splits = {word: list(word) for word in word_freqs.keys()}

        # Learn merges until vocab_size is reached
        while len(self.vocab) < vocab_size:
            pair_freqs = self._get_pair_freqs(word_splits, word_freqs)
            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            merged_token = best_pair[0] + best_pair[1]

            # Add merge rule and new token to vocab
            self.merges[best_pair] = merged_token
            self.vocab.append(merged_token)

            # Apply merge to all word splits
            word_splits = self._merge_pair(word_splits, best_pair)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text using learned merges."""
        words = text.lower().split()
        tokens = []

        for word in words:
            # Split word into characters
            word_tokens = list(word)

            # Apply merges in order
            for pair, merged in self.merges.items():
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        word_tokens = word_tokens[:i] + [merged] + word_tokens[i + 2:]
                    else:
                        i += 1

            tokens.extend(word_tokens)

        return tokens

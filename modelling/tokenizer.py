import os
import json
import tempfile
from collections import Counter
from typing import List, Dict, Tuple, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class BPETokenizer:
    """Byte Pair Encoding tokenizer implementation (from scratch)."""

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
        word_freqs = self._get_word_freqs(corpus)

        # Build base vocabulary (unique characters)
        base_vocab = set()
        for word in word_freqs.keys():
            for char in word:
                base_vocab.add(char)
        self.vocab = sorted(list(base_vocab))

        # Initialize word splits
        word_splits = {word: list(word) for word in word_freqs.keys()}

        # Learn merges until vocab_size is reached
        while len(self.vocab) < vocab_size:
            pair_freqs = self._get_pair_freqs(word_splits, word_freqs)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            merged_token = best_pair[0] + best_pair[1]

            self.merges[best_pair] = merged_token
            self.vocab.append(merged_token)

            word_splits = self._merge_pair(word_splits, best_pair)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text using learned merges."""
        words = text.lower().split()
        tokens = []

        for word in words:
            word_tokens = list(word)

            for pair, merged in self.merges.items():
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        word_tokens = word_tokens[:i] + [merged] + word_tokens[i + 2:]
                    else:
                        i += 1

            tokens.extend(word_tokens)

        return tokens


class HuggingFaceBPETokenizer:
    """
    BPE Tokenizer using HuggingFace's tokenizers library.

    Learns vocabulary using BPETokenizer, then converts to GPT2Tokenizer format
    as specified in Practical 4.
    """

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.tokenizer: Optional[GPT2Tokenizer] = None
        self._base_tokenizer: Optional[Tokenizer] = None

    def train(self, texts: List[str], save_dir: Optional[str] = None):
        """
        Train BPE tokenizer on texts and convert to GPT2Tokenizer.

        Args:
            texts: List of text strings to train on
            save_dir: Directory to save tokenizer files (optional)
        """
        # Initialize BPE tokenizer from HuggingFace tokenizers
        # Use ByteLevel pre-tokenizer to preserve word boundaries (spaces encoded as Ġ)
        self._base_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self._base_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._base_tokenizer.decoder = ByteLevelDecoder()

        # Train with special tokens
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
            show_progress=True
        )

        # Train on texts
        self._base_tokenizer.train_from_iterator(texts, trainer=trainer)

        # Convert to GPT2Tokenizer format
        self._convert_to_gpt2_tokenizer(save_dir)

    def _convert_to_gpt2_tokenizer(self, save_dir: Optional[str] = None):
        """Convert the trained BPE tokenizer to GPT2Tokenizer format."""
        from transformers import PreTrainedTokenizerFast

        if save_dir is None:
            save_dir = tempfile.mkdtemp()

        os.makedirs(save_dir, exist_ok=True)

        # Save the base tokenizer to a JSON file
        tokenizer_json_path = os.path.join(save_dir, "tokenizer.json")
        self._base_tokenizer.save(tokenizer_json_path)

        # Create a PreTrainedTokenizerFast from the tokenizer.json
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_json_path,
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]"
        )

        # Save in the standard format for later loading
        self.tokenizer.save_pretrained(save_dir)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")

        ids = self.tokenizer.encode(text, add_special_tokens=False)

        if add_special_tokens:
            ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to tokens."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.tokenizer.tokenize(text)

    @property
    def vocab_size_actual(self) -> int:
        """Return actual vocabulary size."""
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer)

    @property
    def pad_idx(self) -> int:
        return self.tokenizer.pad_token_id if self.tokenizer else 0

    @property
    def bos_idx(self) -> int:
        return self.tokenizer.bos_token_id if self.tokenizer else 1

    @property
    def eos_idx(self) -> int:
        return self.tokenizer.eos_token_id if self.tokenizer else 2

    @property
    def unk_idx(self) -> int:
        return self.tokenizer.unk_token_id if self.tokenizer else 3

    def save(self, save_dir: str):
        """Save tokenizer to directory."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        os.makedirs(save_dir, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)

    @classmethod
    def load(cls, save_dir: str) -> 'HuggingFaceBPETokenizer':
        """Load tokenizer from directory using from_pretrained."""
        from transformers import PreTrainedTokenizerFast

        instance = cls()

        # Monkeypatch to bypass repo ID validation for local paths
        import huggingface_hub.utils._validators as hf_validators
        original_validate = hf_validators.validate_repo_id
        def patched_validate(repo_id):
            if os.path.exists(repo_id):
                return  # Skip validation for local paths
            return original_validate(repo_id)
        hf_validators.validate_repo_id = patched_validate
        try:
            instance.tokenizer = PreTrainedTokenizerFast.from_pretrained(save_dir)
        finally:
            hf_validators.validate_repo_id = original_validate
        return instance

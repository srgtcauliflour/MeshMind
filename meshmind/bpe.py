"""Trainable byte-level BPE tokenizer for MeshMind."""

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

SPECIAL_TOKENS = ("<bos>", "<eos>", "<pad>")


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = list(merges or [])
        self.merge_ranks = {tuple(pair): rank for rank, pair in enumerate(self.merges)}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(SPECIAL_TOKENS)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            candidates = [
                (self.merge_ranks.get((tokens[i], tokens[i + 1])), i)
                for i in range(len(tokens) - 1)
            ]
            candidates = [(rank, i) for rank, i in candidates if rank is not None]
            if not candidates:
                break
            _, i = min(candidates)
            pair = (tokens[i], tokens[i + 1])
            token_id = 256 + self.merge_ranks[pair]
            tokens[i : i + 2] = [token_id]
        return tokens

    def _expand(self, token):
        if token < 256:
            return [token]
        pair = self.merges[token - 256]
        return self._expand(pair[0]) + self._expand(pair[1])

    def decode(self, ids):
        raw = []
        for token in ids:
            if 0 <= token < 256 + len(self.merges):
                raw.extend(self._expand(token))
        return bytes(raw).decode("utf-8", errors="replace")

    def save(self, path):
        payload = {"version": 1, "merges": self.merges, "special_tokens": SPECIAL_TOKENS}
        Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(pair) for pair in payload["merges"]])


def train_bpe(texts, vocab_size=16_384):
    if vocab_size < 259:
        raise ValueError("vocab_size must leave room for bytes and special tokens")
    sequences = [list(text.encode("utf-8")) for text in texts if text]
    merges = []
    target_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
    for merge_index in range(target_merges):
        counts = Counter()
        for sequence in sequences:
            counts.update(pairwise(sequence))
        if not counts:
            break
        pair, _ = counts.most_common(1)[0]
        new_id = 256 + merge_index
        merges.append(pair)
        for sequence in sequences:
            i = 0
            while i < len(sequence) - 1:
                if (sequence[i], sequence[i + 1]) == pair:
                    sequence[i : i + 2] = [new_id]
                else:
                    i += 1
    return BPETokenizer(merges)

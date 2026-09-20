"""Small, auditable byte-level tokenizer used to bootstrap MM-0A.

This is intentionally a baseline tokenizer. The 16,384-token research tokenizer
will be trained separately once the corpus pipeline is established.
"""

BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
BASE_VOCAB_SIZE = 259


class ByteTokenizer:
    vocab_size = BASE_VOCAB_SIZE

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int]) -> str:
        payload = bytes(token for token in ids if 0 <= token < 256)
        return payload.decode("utf-8", errors="replace")

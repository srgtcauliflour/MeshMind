from meshmind.tokenizer import BOS_ID, EOS_ID, ByteTokenizer


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "MeshMind: distributed intelligence 🧠"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_are_optional():
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("mesh", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID

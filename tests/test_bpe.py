from meshmind.bpe import BPETokenizer, train_bpe


def test_bpe_round_trip_and_compression(tmp_path):
    corpus = ["mesh mesh mesh mind mind", "mesh intelligence"]
    tokenizer = train_bpe(corpus, vocab_size=280)
    text = "mesh mind"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert len(ids) < len(text.encode("utf-8"))

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.encode(text) == ids
    assert restored.decode(ids) == text

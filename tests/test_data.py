from meshmind.data import DataSource, load_manifest, make_causal_blocks, write_manifest


def test_manifest_round_trip(tmp_path):
    source = DataSource("fixture", "local://fixture.txt", "CC0-1.0", "abc123", "test only")
    path = tmp_path / "manifest.json"
    write_manifest(path, [source])
    assert load_manifest(path) == [source]


def test_causal_blocks_shift_targets():
    blocks = list(make_causal_blocks(list(range(10)), context_length=4))
    assert blocks[0] == ([0, 1, 2, 3], [1, 2, 3, 4])
    assert blocks[1] == ([4, 5, 6, 7], [5, 6, 7, 8])

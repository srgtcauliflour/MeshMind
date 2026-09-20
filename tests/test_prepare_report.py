from scripts.prepare_mm_e001 import stats


def test_prep_stats_measure_compression():
    result = stats("hello", [1, 2])
    assert result["bytes"] == 5
    assert result["characters"] == 5
    assert result["tokens"] == 2
    assert result["bytes_per_token"] == 2.5

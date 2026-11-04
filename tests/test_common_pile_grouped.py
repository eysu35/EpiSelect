import os

import pytest

from src.tfds.common_pile_grouped import SPLITS, TASKS, tokenize_capped


def fake_tokenize(texts):
    return [[0] * len(t) for t in texts]


def test_tokenize_capped_stops_at_cap():
    texts = ["a" * 10] * 100
    out = list(tokenize_capped(texts, fake_tokenize, cap=95, batch=8))
    assert [i for i, _ in out] == list(range(10))
    assert sum(len(t) for _, t in out) == 100  # crosses the cap by at most one doc


def test_tokenize_capped_skips_empty_and_flushes_tail():
    texts = ["", "ab", "", "cde", "f"]
    out = list(tokenize_capped(texts, fake_tokenize, cap=10**9, batch=64))
    assert [len(t) for _, t in out] == [2, 3, 1]


def test_split_layout():
    assert len(TASKS) == 31
    assert SPLITS == {"train": "train[1%:]", "val": "train[:1%]"}


def test_train_stream_is_seeded():
    pytest.importorskip("tensorflow")
    data_dir = os.environ.get("EPISELECT_TEST_DATA_DIR", "/path/to/commonpile_tfds")
    if not os.path.isdir(os.path.join(data_dir, "common_pile_grouped", "python_peps")):
        pytest.skip("python_peps not built")
    from src.dataloader.commonpile_loader import make_tfds_dataset

    def first(seed):
        ds = make_tfds_dataset("python_peps", "train", 64, 1, 0, shuffle=True, data_dir=data_dir, seed=seed)
        return [x.tolist() for x in ds.take(8).as_numpy_iterator()]

    assert first(1) == first(1)
    assert first(1) != first(2)


def test_token_counts_sidecar():
    pytest.importorskip("tensorflow")
    import tensorflow_datasets as tfds

    from src.tfds.common_pile_grouped import TOKEN_COUNTS_FILE, read_token_counts, token_counts_path

    data_dir = os.environ.get("EPISELECT_TEST_DATA_DIR", "/path/to/commonpile_tfds")
    builder = tfds.builder("common_pile_grouped/python_peps", data_dir=data_dir)
    if not os.path.exists(token_counts_path(builder)):
        pytest.skip("python_peps not counted")
    assert token_counts_path(builder).endswith(f"python_peps/3.0.0/{TOKEN_COUNTS_FILE}")
    c = read_token_counts(builder)
    assert c["num_examples"] == builder.info.splits["train"].num_examples
    # serialized size is ~2 bytes/token, so the exact count must be well above num_bytes / 4
    assert c["train"] + c["val"] > 1.5 * builder.info.splits["train"].num_bytes / 4
    assert 0 < c["val"] < c["train"]

import os
import pickle

import numpy as np
import pytest

DATA_DIR = os.environ.get("EPISELECT_TEST_DATA_DIR", "/path/to/commonpile_tfds")


def _built(domain):
    return os.path.exists(os.path.join(DATA_DIR, "common_pile_grouped", domain, "3.0.0", "token_counts.json"))


def _selector_kwargs(k):
    return dict(
        n_task=k,
        task_token_count=np.arange(1, k + 1, dtype=np.float64) * 1e6,
        block_size=8,
        ignore_steps=5,
        start_step=10**9,  # never refit; fitting is covered by test_scaling_law
        update_interval=10,
    )


def _drive(sel, steps, seed):
    rng = np.random.default_rng(seed)
    for itr in range(steps):
        ids = rng.integers(0, sel.k, size=16)
        sel.update(ids, rng.random(16), itr, lr=1e-3, metrics={})
        if itr % 5 == 0:
            sel.update_val_loss(rng.random(32), rng.integers(0, sel.k, size=32), itr)


def _assert_same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        if sa[key] is None or isinstance(sa[key], (int, float, str, bool)):
            assert sa[key] == sb[key], key
        else:
            np.testing.assert_array_equal(np.asarray(sa[key]), np.asarray(sb[key]), err_msg=key)


@pytest.mark.parametrize("policy", ["ado", "episelect"])
def test_selector_state_roundtrip(policy):
    from src.dataloader.adaptive_loader import SELECTORS

    kwargs = _selector_kwargs(3)
    if policy == "episelect":
        kwargs.update(
            use_prior=False,
            use_credit_assignment=False,
            do_global_avg=True,
            ema=0.9,
            fit_freq=2,
            window_length=5,
            polyorder=1,
            tau=1.0,
        )
    a = SELECTORS[policy](**kwargs)
    _drive(a, 20, seed=0)
    blob = pickle.dumps(a.state_dict())

    b = SELECTORS[policy](**kwargs)
    b.load_state_dict(pickle.loads(blob))
    _assert_same_state(a, b)
    assert "mesh" not in a.state_dict()

    # both continue identically
    _drive(a, 5, seed=1)
    _drive(b, 5, seed=1)
    _assert_same_state(a, b)


def _stream(domain, seed, skip=0, n=100):
    from src.dataloader.commonpile_loader import make_tfds_dataset

    ds = make_tfds_dataset(domain, "train", 64, 1, 0, shuffle=True, data_dir=DATA_DIR, seed=seed, skip=skip)
    return [x.tolist() for x in ds.take(n).as_numpy_iterator()]


def test_skip_resumes_the_same_stream():
    domain = next((d for d in ("libretexts", "news", "python_peps") if _built(d)), None)
    if domain is None:
        pytest.skip("no test domain built")
    full = _stream(domain, seed=3, n=120)
    assert full == _stream(domain, seed=3, n=120)
    assert _stream(domain, seed=3, skip=50, n=70) == full[50:]


def test_mixture_skip_is_deterministic():
    import tensorflow as tf

    from src.dataloader.commonpile_loader import make_tfds_dataset

    domains = [d for d in ("python_peps", "foodista", "public_domain_review", "libretexts") if _built(d)][:2]
    if len(domains) < 2:
        pytest.skip("need two built domains")

    def mixture(skip):
        dsets = [
            make_tfds_dataset(d, "train", 64, 1, 0, shuffle=True, data_dir=DATA_DIR, seed=i).map(
                lambda x, _i=i: (_i, x)
            )
            for i, d in enumerate(domains)
        ]
        ds = tf.data.Dataset.sample_from_datasets(
            dsets, stop_on_empty_dataset=True, rerandomize_each_iteration=True, seed=7
        )
        return [(int(i), x.tolist()) for i, x in ds.skip(skip).take(80 - skip).as_numpy_iterator()]

    full = mixture(0)
    assert full == mixture(0)
    assert mixture(30) == full[30:]


class _FakeLoader:
    def __init__(self):
        self.host = {"consumed": [1, 2, 3]}
        self.glob = {"cnts": {0: 5}}

    def state_dict(self):
        return {"host": self.host, "global": self.glob}

    def load_state_dict(self, host, glob):
        self.host, self.glob = host, glob


def test_loader_state_files_roundtrip(tmp_path):
    from src.train import load_loader_state, save_loader_state

    src = _FakeLoader()
    save_loader_state(src, str(tmp_path), 1000)
    assert sorted(os.listdir(tmp_path / "loader_state" / "1000")) == ["global.pkl", "host0.pkl"]

    dst = _FakeLoader()
    dst.host, dst.glob = None, None
    load_loader_state(dst, str(tmp_path), 1000)
    assert dst.host == src.host and dst.glob == src.glob

    with pytest.raises(RuntimeError, match="no loader state"):
        load_loader_state(dst, str(tmp_path), 2000)


def test_selector_state_keeps_python_scalars():
    from src.data_selectors.ado import AdoSelector

    state = AdoSelector(**_selector_kwargs(3)).state_dict()
    assert type(state["k"]) is int and type(state["y_axis"]) is str and type(state["use_prior"]) is bool
    assert state["current_params"] is None


def test_done_marker(tmp_path):
    import json

    from src.train import done_path, write_done

    write_done(str(tmp_path), 59999)
    assert json.load(open(done_path(str(tmp_path)))) == {"step": 59999}


def test_resume_step_prefers_newest_with_loader_state(tmp_path):
    from src.train import resume_step, save_loader_state

    class FakeLoader:
        def state_dict(self):
            return {"host": {"h": 1}, "global": {"g": 2}}

    rundir = str(tmp_path)
    save_loader_state(FakeLoader(), rundir, 1000)
    save_loader_state(FakeLoader(), rundir, 2000)
    assert resume_step(rundir, [1000, 2000, 2731]) == 2000  # 2731: preemption save without loader state
    assert resume_step(rundir, []) is None
    with pytest.raises(RuntimeError):
        resume_step(rundir, [2731])

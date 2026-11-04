import json

from scripts.aggregate_eval import aggregate, flatten, markdown_tables, mean_token_stats


def _rd(label, seed, val, c4, acc):
    return {
        "label": label,
        "seed": seed,
        "rundir": f"gs://b/{label}/seed{seed}",
        "step": 59999,
        "val": {"empirical_loss": val, "balanced_loss": val + 0.1, "loss/task0": 1.0, "eval_time": 3.0},
        "c4": {"loss": c4, "perplexity": 2.0},
        "harness": {"piqa": {"acc": acc, "stderr": 0.01}},
    }


def test_aggregate_over_seeds():
    results = {
        "runs": {
            "ado_seed0": _rd("ADO", 0, 3.0, 4.0, 0.6),
            "ado_seed1": _rd("ADO", 1, 3.2, 4.4, 0.7),
            "nat": _rd("Natural", None, 3.5, 4.5, 0.5),
        }
    }
    agg = aggregate(results)
    assert (
        agg["ADO"]["val.empirical_loss"] == {"mean": 3.1, "std": 0.1414213562373099, "n": 2}
        or abs(agg["ADO"]["val.empirical_loss"]["std"] - 0.14142) < 1e-4
    )
    assert agg["ADO"]["harness.piqa.acc"]["n"] == 2 and "harness.piqa.stderr" not in agg["ADO"]
    assert "seed" not in agg["ADO"] and "step" not in agg["ADO"]
    assert agg["Natural"]["c4.loss"] == {"mean": 4.5, "std": 0.0, "n": 1}

    table = markdown_tables(agg, labels=["Natural", "ADO"])
    assert "| val.empirical_loss | 3.5000 ± 0.0000 (n=1) | 3.1000 ±" in table
    assert "val.loss/task0" not in table and "### harness" in table


def test_flatten_skips_non_numeric():
    assert flatten({"a": {"b": 1, "c": "x", "d": None, "e": True}, "f": 2.5}) == {"a.b": 1.0, "f": 2.5}


def test_mean_token_stats(tmp_path):
    for i, tot in enumerate((100, 300)):
        d = tmp_path / f"seed{i}"
        d.mkdir()
        json.dump(
            {"step": 1, "total_tokens": tot, "per_domain": {"a": tot, "b": 0 if i else 10}},
            open(d / "token_stats.json", "w"),
        )
    m = mean_token_stats([str(tmp_path / "seed0"), str(tmp_path / "seed1"), str(tmp_path / "seed2")])
    assert m == {"n_seeds": 2, "total_tokens": 200.0, "per_domain": {"a": 200.0, "b": 5.0}}


def test_group_averages_and_harness_table():
    from scripts.aggregate_eval import add_group_averages, harness_table

    groups = {"all": ["a", "b", "c"], "sub": ["a", "b"]}
    results = {
        "runs": {
            "m_seed0": {
                "label": "M",
                "seed": 0,
                "harness": {t: {"acc": v, "stderr": 0.1} for t, v in zip("abc", (0.2, 0.4, 0.9))},
            },
            "m_seed1": {
                "label": "M",
                "seed": 1,
                "harness": {"a": {"acc": 0.4, "stderr": 0.1}, "b": {"acc": 0.6, "stderr": 0.1}},
            },
        }
    }
    add_group_averages(results, groups, merged={"ab": ["a", "b"]})
    assert results["runs"]["m_seed0"]["harness"]["ab"] == {"acc": 0.30000000000000004}
    assert "ab" not in results["runs"]["m_seed1"]["harness"] or results["runs"]["m_seed1"]["harness"]["ab"]["acc"] == 0.5
    assert results["runs"]["m_seed0"]["harness_avg"] == {"all": 0.5, "sub": 0.30000000000000004}
    assert results["runs"]["m_seed1"]["harness_avg"] == {"sub": 0.5}
    agg = aggregate(results)
    assert agg["M"]["harness_avg.sub"]["n"] == 2 and agg["M"]["harness_avg.all"]["n"] == 1
    table = harness_table(agg, groups, merged={"ab": ["a", "b"]})
    assert "avg sub" in table and "0.400 ± 0.141" in table and "| ab |" in table

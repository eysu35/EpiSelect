import argparse
import json
import os
from collections import defaultdict

import gcsfs
import numpy as np
import yaml

EVAL_CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "eval.yaml")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
            out[key] = float(v)
    return out


SKIP = {"seed", "step", "harness_time"}


def harness_groups(path=EVAL_CFG):
    cfg = yaml.safe_load(open(path))
    return {"all": list(cfg["harness_tasks"]), **cfg.get("harness_groups", {})}, cfg.get("harness_merged", {})


def add_group_averages(results, groups, merged=None):
    # merged tasks (e.g. arc = mean of arc_challenge/arc_easy) become harness entries; then per-run average over each
    # task group; a run missing any task of a group gets no value for it
    for rd in results["runs"].values():
        h = rd.get("harness", {})
        for name, parts in (merged or {}).items():
            if all(h.get(t, {}).get("acc") is not None for t in parts):
                h[name] = {"acc": float(np.mean([h[t]["acc"] for t in parts]))}
        accs = {t: r["acc"] for t, r in h.items() if r.get("acc") is not None}
        avg = {g: float(np.mean([accs[t] for t in ts])) for g, ts in groups.items() if all(t in accs for t in ts)}
        if avg:
            rd["harness_avg"] = avg


def aggregate(results):
    # {label: {metric: {"mean", "std", "n"}}} over seeds; unseeded runs count as one seed
    per_label = defaultdict(lambda: defaultdict(list))
    for rd in results["runs"].values():
        for k, v in flatten(rd).items():
            if k not in SKIP and not k.endswith(".stderr"):
                per_label[rd["label"]][k].append(v)
    return {
        label: {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
            for k, v in ms.items()
        }
        for label, ms in per_label.items()
    }


def markdown_tables(agg, labels=None):
    labels = labels or list(agg)
    metrics = sorted({m for label in labels for m in agg[label]})
    groups = defaultdict(list)
    for m in metrics:
        groups[m.split(".")[0]].append(m)
    lines = []
    for group, ms in groups.items():
        if group == "val":
            ms = [
                m for m in ms if not m.startswith("val.loss/task") and m != "val.eval_time" and m != "val.tot_val_exs"
            ]
        lines.append(f"\n### {group}\n")
        lines.append("| metric | " + " | ".join(labels) + " |")
        lines.append("|---|" + "---|" * len(labels))
        for m in ms:
            cells = []
            for label in labels:
                s = agg[label].get(m)
                cells.append("" if s is None else f"{s['mean']:.4f} ± {s['std']:.4f} (n={s['n']})")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def harness_table(agg, groups, labels=None, merged=None):
    # methods as rows; one column per task and merged task, then one per group average
    labels = labels or list(agg)
    tasks = groups["all"] + list(merged or {})
    cols = [(t, f"harness.{t}.acc") for t in tasks] + [(f"avg {g}", f"harness_avg.{g}") for g in groups]
    lines = ["\n### harness (acc, mean ± std over seeds)\n", "| method | n | " + " | ".join(c for c, _ in cols) + " |"]
    lines.append("|---|---|" + "---|" * len(cols))
    for label in labels:
        n = max((agg[label][m]["n"] for _, m in cols if m in agg[label]), default=0)
        cells = [
            f"{agg[label][m]['mean']:.3f} ± {agg[label][m]['std']:.3f}" if m in agg[label] else "" for _, m in cols
        ]
        lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _open(path, mode="r"):
    return gcsfs.GCSFileSystem().open(path, mode) if path.startswith("gs://") else open(path, mode)


def mean_token_stats(rundirs):
    # average token_stats.json over seeds for the mixture plots
    stats = []
    for rd in rundirs:
        try:
            with _open(os.path.join(rd, "token_stats.json")) as f:
                stats.append(json.load(f))
        except FileNotFoundError:
            print(f"missing token_stats.json in {rd}")
    if not stats:
        return None
    domains = sorted({d for s in stats for d in s["per_domain"]})
    return {
        "n_seeds": len(stats),
        "total_tokens": float(np.mean([s["total_tokens"] for s in stats])),
        "per_domain": {d: float(np.mean([s["per_domain"].get(d, 0) for s in stats])) for d in domains},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="eval_results json(s) written by src.eval; runs are merged")
    parser.add_argument("--out", help="write the aggregate json here (default: <first results>_agg.json)")
    parser.add_argument(
        "--token_stats", action="store_true", help="also average token_stats.json over each label's rundirs"
    )
    args = parser.parse_args()

    results = {"runs": {}}
    for path in args.results:
        results["runs"].update(json.load(open(path)).get("runs", {}))
    groups, merged = harness_groups()
    add_group_averages(results, groups, merged)
    agg = aggregate(results)
    print(markdown_tables(agg))
    print(harness_table(agg, groups, merged=merged))

    out = {"metrics": agg}
    if args.token_stats:
        by_label = defaultdict(list)
        for rd in results["runs"].values():
            by_label[rd["label"]].append(rd["rundir"])
        out["token_stats"] = {label: mean_token_stats(rds) for label, rds in by_label.items()}

    out_path = args.out or args.results[0].replace(".json", "_agg.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()

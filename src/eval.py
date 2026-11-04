# offline eval of trained checkpoints: common pile val, an OOD token stream, and the
# lm eval harness. results accumulate in cfg.output and finished runs are skipped

import json
import os
import time

import equinox as eqx
import gcsfs
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
from datasets import load_dataset
from jax.experimental import mesh_utils
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from src.dataloader.commonpile_loader import make_tfds_dataset
from src.harness.lm_harness_eval import LMEvaluator
from src.model import GPT, count_params, shard_gpt
from src.sharding import reshard
from src.tfds import common_pile_grouped
from src.train import cast_pytree, make_training_fns

jtu = jax.tree_util
NamedSharding, P = jax.sharding.NamedSharding, jax.sharding.PartitionSpec

# lm-eval's bootstrap forks the JAX process once per core (host OOM)
os.environ.setdefault("DISABLE_MULTIPROC", "1")


def _read_json(path):
    if path.startswith("gs://"):
        with gcsfs.GCSFileSystem().open(path) as f:
            return json.load(f)
    return json.load(open(path))


def _slug(label):
    return label.lower().replace(" ", "_").replace("-", "_")


def expand_runs(runs):
    # (key, label, seed, rundir) per seed; an entry without `seeds` is a single unseeded rundir
    out = []
    for r in runs:
        seeds = r.get("seeds")
        if seeds is None:
            out.append((_slug(r["label"]), r["label"], None, r["path"]))
        else:
            out.extend((f"{_slug(r['label'])}_seed{s}", r["label"], int(s), f"{r['path']}/seed{s}") for s in seeds)
    return out


def _rundir_exists(rundir):
    # os.path.isdir is always False for gs:// URIs, which _read_json does support
    if rundir.startswith("gs://"):
        return gcsfs.GCSFileSystem().exists(rundir)
    return os.path.isdir(rundir)


def _namespace(d):
    # rebuild the training config from cfg.json as an attribute-access object
    ns = type("C", (), {})()
    for k, v in d.items():
        setattr(ns, k, _namespace(v) if isinstance(v, dict) else v)
    return ns


def _load_model(rundir, mesh, cfg):
    model = GPT(cfg.model, jax.random.PRNGKey(0))
    model = cast_pytree(model, cfg.param_dtype)
    model = shard_gpt(model, mesh, cfg.shard_model)
    print(f"  {count_params(model):,} params")

    # The optimizer state is part of the checkpoint, so rebuild it to restore
    sched = optax.warmup_cosine_decay_schedule(0, cfg.learning_rate, cfg.warmup_steps, cfg.lr_decay_steps, cfg.min_lr)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b2=cfg.beta2),
        optax.add_decayed_weights(cfg.weight_decay / cfg.learning_rate),
        optax.scale_by_schedule(sched),
        optax.scale(-1),
    )
    opt_state = jtu.tree_map(
        lambda x: reshard(x, NamedSharding(mesh, P())) if eqx.is_array(x) and x.ndim == 0 else x,
        opt.init(eqx.filter(model, eqx.is_array)),
    )
    mngr = ocp.CheckpointManager(
        rundir, ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()), ocp.CheckpointManagerOptions(max_to_keep=1)
    )
    step = mngr.latest_step()
    assert step is not None, f"No checkpoint in {rundir}"

    ex = (jtu.tree_leaves(model), jtu.tree_leaves(opt_state))
    restore_args = ocp.checkpoint_utils.construct_restore_args(
        ex, jtu.tree_map(lambda x: x.sharding if eqx.is_array(x) else None, ex)
    )
    leaves, _ = mngr.restore(step, restore_kwargs={"restore_args": list(restore_args)})
    return jtu.tree_unflatten(jtu.tree_structure(model), leaves), step


def _commonpile_dset(split, batch_size, block_size, data_dir):
    n, idx = jax.process_count(), jax.process_index()
    dsets = [
        make_tfds_dataset(t, split, block_size + 1, n, idx, shuffle=False, data_dir=data_dir, repeat=False).map(
            lambda x, _i=i: (_i, x)
        )
        for i, t in enumerate(common_pile_grouped.TASKS)
    ]
    return (
        tf.data.Dataset.sample_from_datasets(
            dsets, stop_on_empty_dataset=False, rerandomize_each_iteration=False, seed=0
        )
        .batch(batch_size // n, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def _eval_commonpile(model, eval_fn, split, data_dir, batch_size, block_size, val_batch_size=None):
    # same 150 x batch_size sequences as the in-training eval, in chunks that fit one GPU
    vb = val_batch_size or batch_size
    _, _, m = eval_fn(model, _commonpile_dset(split, vb, block_size, data_dir), max_batches=150 * batch_size // vb)
    return {k: float(v) for k, v in m.items()}


def _get_hf_tokens(hf_path, subset, split, text_field, cache_path, max_tokens=10_000_000):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    kwargs = dict(split=split, streaming=True, trust_remote_code=True)
    if subset is not None:
        kwargs["name"] = subset
    ds = load_dataset(hf_path, **kwargs)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokens = []
    for ex in ds:
        tokens.extend(tok.encode(ex[text_field]))
        if len(tokens) >= max_tokens:
            break
    arr = np.array(tokens[:max_tokens], dtype=np.int32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, arr)
    return arr


OOD_SOURCES = {
    "c4": ("allenai/c4", "en", "validation", "text"),
    "slim_pajama": ("gmongaras/SlimPajama-627B_Reupload", None, "validation", "text"),
    "fine_web": ("HuggingFaceFW/fineweb", "sample-10BT", "train", "text"),
}


def _get_ood_tokens(name, cache_dir, max_tokens=10_000_000):
    hf_path, subset, split, field = OOD_SOURCES[name]
    cache_path = os.path.join(cache_dir, f"{name}_tokens_{max_tokens // 1_000_000}M.npy")
    return _get_hf_tokens(hf_path, subset, split, field, cache_path, max_tokens)


@eqx.filter_jit
def _token_loss(model, x):
    lp = jax.nn.log_softmax(jax.vmap(model)(x[:, :-1], key=None), axis=-1)
    return -jnp.take_along_axis(lp, x[:, 1:, None], axis=-1).squeeze(-1).mean()


def _eval_tokens(model, tokens, block_size, batch_size, mesh):
    sharding = NamedSharding(mesh, P("data", None))
    sl = block_size + 1
    seqs = tokens[: (len(tokens) // sl) * sl].reshape(-1, sl)
    losses = [
        float(_token_loss(model, jax.device_put(jnp.array(seqs[i * batch_size : (i + 1) * batch_size]), sharding)))
        for i in range(len(seqs) // batch_size)
    ]
    loss = float(np.mean(losses))
    return {"loss": loss, "perplexity": float(np.exp(loss))}


def _eval_harness(model, mesh, tasks, batch_size, limit, out, save):
    # out holds the tasks already evaluated; saved after every task
    for task in tasks:
        if task in out:
            continue
        print(f"\nEvaluating: {task}...")
        tic = time.time()
        results = simple_evaluate(
            model=LMEvaluator(model, mesh, batch_size=batch_size),
            tasks=[task],
            num_fewshot=0,
            task_manager=TaskManager(),
            limit=limit,
            bootstrap_iters=1000,
            confirm_run_unsafe_code=True,
        )
        r = results["results"].get(task, {})
        acc = r.get("acc,none", r.get("acc_norm,none"))
        stderr = r.get("acc_stderr,none", r.get("acc_norm_stderr,none"))
        print(f"{task} done in {time.time() - tic:.1f}s — acc: {acc:.2f}")
        out[task] = {"acc": acc, "stderr": stderr}
        save()


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    runs = expand_runs(OmegaConf.to_container(cfg.runs, resolve=True))
    if cfg.get("labels"):
        runs = [r for r in runs if r[1] in cfg.labels]
    evals, tasks = set(cfg.evals), list(cfg.harness_tasks)

    local = jax.local_device_count()
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh((jax.device_count() // local, local)), ("replica", "data"))

    # make_training_fns needs an optimizer to build its closures, but eval never steps it
    dummy_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b2=0.95),
        optax.add_decayed_weights(0.0),
        optax.scale_by_schedule(lambda _: 0.0),
        optax.scale(-1),
    )

    ood_tokens = {name: _get_ood_tokens(name, cfg.ood_cache_dir) for name in OOD_SOURCES if name in evals}

    results = {"evals": list(evals), "runs": {}}
    if os.path.exists(cfg.output):
        results["runs"].update(json.load(open(cfg.output)).get("runs", {}))

    def save():
        os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)
        json.dump(results, open(cfg.output, "w"), indent=2)

    for i, (key, label, seed, rundir) in enumerate(runs):
        print(f"\n[{i + 1}/{len(runs)}] {label} seed {seed}")
        rd = results["runs"].get(key, {})
        todo = [e for e in evals - {"harness"} if e not in rd]
        todo_tasks = [t for t in tasks if t not in rd.get("harness", {})] if "harness" in evals else []
        if not todo and not todo_tasks:
            print("  already done, skipping")
            continue
        if not _rundir_exists(rundir):
            print(f"  ERROR: not found: {rundir}")
            continue

        run_cfg = _namespace(_read_json(f"{rundir}/cfg.json"))
        model, step = _load_model(rundir, mesh, run_cfg)
        _, eval_fn = make_training_fns(run_cfg, dummy_opt, mesh)
        rd.update({"label": label, "seed": seed, "rundir": rundir, "step": int(step)})
        results["runs"][key] = rd

        # val is the held-out 1% slice of the built stream (src/tfds/common_pile_grouped.py)
        if "val" in todo:
            rd["val"] = _eval_commonpile(
                model, eval_fn, "val", cfg.data_dir, run_cfg.batch_size, run_cfg.block_size, cfg.get("val_batch_size")
            )

        inf_model = eqx.Partial(cast_pytree(model, jnp.float32), inference=True)
        for name, tokens in ood_tokens.items():
            if name in todo:
                rd[name] = _eval_tokens(inf_model, tokens, run_cfg.block_size, cfg.ood_batch_size, mesh)
        save()

        if todo_tasks:
            t0 = time.time()
            _eval_harness(model, mesh, tasks, cfg.batch_size, cfg.harness_limit, rd.setdefault("harness", {}), save)
            rd["harness_time"] = rd.get("harness_time", 0.0) + time.time() - t0

        del model
        save()
        print(f"  saved → {cfg.output}")


if __name__ == "__main__":
    main()

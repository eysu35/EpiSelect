import json
import os
import pickle
import time
import typing as tp
from functools import partial

import equinox as eqx
import gcsfs
import hydra
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import wandb
from jax.experimental import mesh_utils, multihost_utils
from tqdm import trange

from src.dataloader import commonpile_loader
from src.dataloader.abstract import Loader

from .model import GPT, count_params, shard_gpt
from .sharding import get_data_sharding, get_shard_fn, reshard

jax.config.update("jax_threefry_partitionable", True)

jnp, jrandom, vmap, scan, jtu = jax.numpy, jax.random, jax.vmap, jax.lax.scan, jax.tree_util
Array = jax.Array
KeyArray = tp.Any
Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def cast_pytree(pytree: tp.Any, dtype: jnp.dtype) -> tp.Any:
    # cast every array leaf to dtype, leaving non-arrays untouched

    def cast(x):
        return x.astype(dtype) if eqx.is_array(x) else x

    return jtu.tree_map(cast, pytree)


def make_training_fns(
    config, optimizer: optax.GradientTransformationExtraArgs, mesh: Mesh
) -> tp.Tuple[tp.Callable, tp.Callable]:
    def loss_fn(
        model_params: GPT,
        model_static: GPT,
        x_BxT: Array,
        y_BxT: Array,
        key: tp.Optional[KeyArray],
    ) -> Array:
        model = eqx.combine(model_params, model_static)
        if key is not None:
            key = jrandom.split(key, x_BxT.shape[0])
        logits_BxTxV = vmap(model)(x_BxT, key=key).astype(jnp.float32)
        vector_loss_BxT = optax.softmax_cross_entropy_with_integer_labels(logits_BxTxV, y_BxT)
        vector_loss_B = vector_loss_BxT.mean(axis=-1)
        return vector_loss_B.mean(), vector_loss_B

    @partial(eqx.filter_jit, donate="all")
    def step(model: GPT, opt_state, x_GxBxT: Array, y_GxBxT: Array, key: KeyArray):
        G = config.g_accum_iters
        params, static = eqx.partition(model, eqx.is_array)
        params_cpt = cast_pytree(params, jnp.dtype(config.compute_dtype))

        def microstep(grad_so_far, xykey_g: tp.Tuple[Array, Array, Array, KeyArray]):
            grad, loss_vector = jax.grad(loss_fn, has_aux=True)(params_cpt, static, *xykey_g)
            grad = shard_gpt(grad, mesh, config.shard_model)
            return jtu.tree_map(lambda x, y: x + y, grad, grad_so_far), loss_vector

        all_keys = jrandom.split(key, G)
        init_grad = jtu.tree_map(jnp.zeros_like, params)
        grad, loss_GxB = scan(microstep, init_grad, (x_GxBxT, y_GxBxT, all_keys))
        # Gradients are summed over the G microbatches, so rescale
        loss, grad = jnp.mean(loss_GxB), jtu.tree_map(lambda x: x / G, grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        model = eqx.combine(optax.apply_updates(params, updates), static)
        return model, opt_state, loss, loss_GxB

    @eqx.filter_jit
    def simple_loss(model, x_1xBxD: Array, y_1xBxD: Array, key: tp.Optional[KeyArray]) -> Array:
        x_BxD, y_BxD = x_1xBxD.squeeze(0), y_1xBxD.squeeze(0)
        model_params, model_static = eqx.partition(model, eqx.is_array)
        return loss_fn(model_params, model_static, x_BxD, y_BxD, key)

    sharding_fn = get_shard_fn(mesh, get_data_sharding(mesh))

    def evaluate(model: GPT, dset, max_batches: int = 150) -> tp.Dict[int, float]:
        start_time = time.time()
        eval_model = eqx.Partial(cast_pytree(model, jnp.dtype(config.compute_dtype)), inference=True)
        metrics = {}
        loss_np_N = np.zeros((0,), dtype=np.float64)
        taskid_np_N = np.zeros((0,), dtype=np.int32)
        # Capped at 150 batches: dataset sharding is uneven across hosts, and letting some
        # workers exhaust their shard before others deadlocks the allgather below
        for taskid_B, seq_BxDp1 in dset.take(max_batches).as_numpy_iterator():
            x_1xBxD, y_1xBxD = seq_BxDp1[:, :-1][None], seq_BxDp1[:, 1:][None]
            x_1xBxD, y_1xBxD, taskid_1xB = jtu.tree_map(sharding_fn, (x_1xBxD, y_1xBxD, taskid_B[None]))
            _, loss_B = simple_loss(eval_model, x_1xBxD, y_1xBxD, None)
            loss_np_B, taskid_np_1xB = multihost_utils.process_allgather((loss_B, taskid_1xB), tiled=True)
            loss_np_N = np.concatenate((loss_np_N, np.asarray(loss_np_B).reshape(-1)))
            taskid_np_N = np.concatenate((taskid_np_N, np.asarray(taskid_np_1xB).reshape(-1)))

        metrics["empirical_loss"] = loss_np_N.mean()
        losses = []
        for task in np.unique(taskid_np_N).tolist():
            task_loss = loss_np_N[taskid_np_N == task].mean()
            metrics[f"loss/task{task}"] = task_loss
            losses.append(task_loss)
        metrics["balanced_loss"] = np.mean(losses)
        metrics["eval_time"] = time.time() - start_time
        metrics["tot_val_exs"] = loss_np_N.shape[0]
        return loss_np_N, taskid_np_N, metrics

    return step, evaluate


def _loader_state_dir(rundir, step):
    return os.path.join(rundir, "loader_state", str(step))


def save_loader_state(train_sampler: Loader, rundir, step):
    # written before the orbax checkpoint of the same step, so any restorable step has loader state
    d = _loader_state_dir(rundir, step)
    tf.io.gfile.makedirs(d)
    state = train_sampler.state_dict()
    with tf.io.gfile.GFile(os.path.join(d, f"host{jax.process_index()}.pkl"), "wb") as f:
        pickle.dump(state["host"], f)
    if jax.process_index() == 0:
        with tf.io.gfile.GFile(os.path.join(d, "global.pkl"), "wb") as f:
            pickle.dump(state["global"], f)


def _loader_state_paths(rundir, step):
    d = _loader_state_dir(rundir, step)
    return [os.path.join(d, f"host{jax.process_index()}.pkl"), os.path.join(d, "global.pkl")]


def has_loader_state(rundir, step):
    return all(tf.io.gfile.exists(p) for p in _loader_state_paths(rundir, step))


def resume_step(rundir, steps):
    # newest checkpoint that also has loader state; orbax's preemption saves land on arbitrary steps
    for step in sorted(steps, reverse=True):
        if has_loader_state(rundir, step):
            return step
    if steps:
        raise RuntimeError(
            f"No checkpoint in {rundir} has loader state (steps {sorted(steps)}); cannot resume without replaying data."
        )
    return None


def load_loader_state(train_sampler: Loader, rundir, step):
    paths = _loader_state_paths(rundir, step)
    missing = [p for p in paths if not tf.io.gfile.exists(p)]
    if missing:
        raise RuntimeError(f"Checkpoint {step} has no loader state ({missing}); cannot resume without replaying data.")
    with tf.io.gfile.GFile(paths[0], "rb") as f:
        host = pickle.load(f)
    with tf.io.gfile.GFile(paths[1], "rb") as f:
        glob = pickle.load(f)
    train_sampler.load_state_dict(host, glob)


def done_path(rundir):
    return os.path.join(rundir, "DONE")


def write_done(rundir, step):
    # marker the run queue uses to skip finished runs
    with tf.io.gfile.GFile(done_path(rundir), "w") as f:
        json.dump({"step": int(step)}, f)


def _save_token_stats(train_sampler, config, itr):
    domain_names = train_sampler.tasks_sampler.tasks
    stats = {
        "step": itr,
        "total_tokens": int(sum(train_sampler.cumulative_cnts.values()) * config.block_size),
        "per_domain": {
            domain_names[task_id] if task_id < len(domain_names) else str(task_id): int(cnt * config.block_size)
            for task_id, cnt in train_sampler.cumulative_cnts.items()
        },
    }
    path = os.path.join(config.rundir, "token_stats.json")
    if config.rundir.startswith("gs://"):
        with gcsfs.GCSFileSystem().open(path, "w") as f:
            json.dump(stats, f, indent=2)
    else:
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)


def train(config):
    n_devices = jax.device_count()
    local_cnt = jax.local_device_count()
    assert local_cnt in {1, 4, 8}, f"Unexpected local device count: {local_cnt}."
    mesh = Mesh(
        mesh_utils.create_device_mesh((n_devices // local_cnt, local_cnt)),
        axis_names=("replica", "data"),
    )

    val_dset = commonpile_loader.make_eval_dset(config.batch_size, config.block_size, data_dir=config.loader.data_dir)

    if not config.debug:
        mngr = ocp.CheckpointManager(
            config.rundir,
            ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
            options=ocp.CheckpointManagerOptions(max_to_keep=None, save_interval_steps=config.eval_interval),
        )

    scheduler = optax.warmup_cosine_decay_schedule(
        0, config.learning_rate, config.warmup_steps, config.lr_decay_steps, end_value=config.min_lr
    )

    @jax.jit
    def get_lr(_opt_state):
        return scheduler(_opt_state[3].count)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b2=config.beta2),
        optax.add_decayed_weights(config.weight_decay / config.learning_rate),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    )
    step, evaluate = make_training_fns(config, optimizer, mesh)

    key = jrandom.PRNGKey(config.seed)

    def init_model(model_key):
        model = GPT(config.model, model_key)
        model = cast_pytree(model, config.param_dtype)
        return shard_gpt(model, mesh, config.shard_model)

    key, key1 = jrandom.split(key)
    model = eqx.filter_jit(init_model)(key1)
    print(f"Model has {count_params(model):,d} parameters.")

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    # Scalars must be explicitly replicated across the mesh
    opt_state = jtu.tree_map(lambda x: reshard(x, NamedSharding(mesh, P())) if x.ndim == 0 else x, opt_state)

    first_step = 0
    start = None if config.debug else resume_step(config.rundir, mngr.all_steps())
    if start is not None:
        ex_state = (jtu.tree_leaves(model), jtu.tree_leaves(opt_state))
        ex_shardings = jtu.tree_map(lambda x: x.sharding if eqx.is_array(x) else None, ex_state)
        abstract = jtu.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) if isinstance(x, jax.Array) else x,
            ex_state,
        )
        restore_args = ocp.checkpoint_utils.construct_restore_args(abstract, ex_shardings)
        model_struct, opt_struct = jtu.tree_structure(model), jtu.tree_structure(opt_state)
        # free the initialized state first: 1.3B does not fit twice in HBM
        for x in jtu.tree_leaves(ex_state):
            if isinstance(x, jax.Array):
                x.delete()
        del model, opt_state, ex_state
        model_leaves, opt_state_leaves = mngr.restore(start, restore_kwargs={"restore_args": restore_args})
        model = jtu.tree_unflatten(model_struct, model_leaves)
        opt_state = jtu.tree_unflatten(opt_struct, opt_state_leaves)
        first_step = start + 1

    train_sampler: Loader = hydra.utils.instantiate(
        config.loader,
        batch_size=config.batch_size,
        split="train",
        mesh=mesh,
        block_size=config.block_size,
        g_accum_iters=config.g_accum_iters,
        seed=config.seed,
    )
    print(f"Training over {train_sampler.tasks_sampler.tasks} domains.")
    if first_step > 0:
        load_loader_state(train_sampler, config.rundir, first_step - 1)
        print(f"Resumed loader state from step {first_step - 1}.")

    pbar = trange(
        first_step,
        config.max_steps,
        initial=first_step,
        total=config.max_steps,
        disable=jax.process_index() != 0,
    )
    is_main = jax.process_index() == 0
    for itr in pbar:
        metrics = {}
        if itr % config.eval_interval == 0:
            loss_np_N, taskid_np_N, val_metrics = evaluate(model, val_dset)
            if hasattr(train_sampler, "update_val_loss"):
                train_sampler.update_val_loss(loss_np_N, taskid_np_N, itr)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            if not config.debug and is_main:
                _save_token_stats(train_sampler, config, itr)

        key, key1 = jrandom.split(key)
        x_GxBxD, y_GxBxD, info = train_sampler.get_batch(itr)
        model, opt_state, loss, loss_GxB = step(model, opt_state, x_GxBxD, y_GxBxD, key1)
        metrics.update(
            {f"opt_{k}": v for k, v in train_sampler.update(loss_GxB, info, itr, lr=get_lr(opt_state).item()).items()}
        )

        if not config.debug:
            if mngr.should_save(itr):  # interval saves and orbax's save-on-preemption alike
                save_loader_state(train_sampler, config.rundir, itr)
            mngr.save(itr, (jtu.tree_leaves(model), jtu.tree_leaves(opt_state)))
        metrics["loss/opt"] = loss.item()
        metrics["lr"] = get_lr(opt_state).item()

        if not config.debug and is_main and itr % 20 == 0:
            if pbar.format_dict["rate"] is not None:
                tok_per_step = config.batch_size * config.g_accum_iters * config.block_size
                metrics["perf/tps"] = pbar.format_dict["rate"] * tok_per_step
            wandb.log(metrics, step=itr)
        pbar.set_postfix(**metrics)

    pbar.close()
    if not config.debug:
        last = config.max_steps - 1
        if mngr.latest_step() != last:  # final checkpoint, whatever the save interval
            save_loader_state(train_sampler, config.rundir, last)
            mngr.save(last, (jtu.tree_leaves(model), jtu.tree_leaves(opt_state)), force=True)
        mngr.wait_until_finished()
        if jax.process_index() == 0:
            _save_token_stats(train_sampler, config, last)
            write_done(config.rundir, last)

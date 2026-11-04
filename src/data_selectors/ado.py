# ADO baseline
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxopt import LBFGS
from scipy.signal import savgol_filter


class AdoSelector:
    def __init__(
        self,
        n_task,
        task_token_count,
        block_size,
        ignore_steps,
        start_step,
        update_interval,
        y_axis="train_loss",  # train_loss | val_loss
        use_same_step_size=True,
        divide_by_n=True,
        use_prior=True,
    ):
        self.k = n_task
        self.block_size = block_size
        self.tokens_remaining = task_token_count
        self.empirical_dist = task_token_count / task_token_count.sum()
        self.remain_dist = self.empirical_dist
        self.ignore_steps = ignore_steps
        self.start_step = start_step
        self.y_axis = y_axis
        self.update_interval = update_interval
        self.use_same_step_size = use_same_step_size
        self.divide_by_n = divide_by_n
        self.use_prior = use_prior
        print(f"Selector warmup={ignore_steps}, start_step={start_step}, update_interval={update_interval}.")

        self.current_params = None
        self.task_cnt_TxK = np.empty((0, self.k), dtype=np.int32)
        self.task_loss_TxK = np.empty((0, self.k), dtype=np.float32)
        self.val_loss_SxK = np.empty((0, self.k), dtype=np.float32)
        self.val_cnt_SxK = np.empty((0, self.k), dtype=np.float64)
        self.val_step_S = np.empty((0,), dtype=np.int32)

        self.p = self.empirical_dist
        self.p_avg = self.empirical_dist
        self.p_history = self.empirical_dist
        self.p_scaling = self.empirical_dist
        self.current_lr = 1e-3

        self.train_epi_ceiling = np.zeros(self.k, dtype=np.float64)
        self.train_total_nk = np.zeros(self.k, dtype=np.float64)
        self.val_epi_ceiling = np.zeros(self.k, dtype=np.float64)
        self.val_total_nk = np.zeros(self.k, dtype=np.float64)

        local_cnt = jax.local_device_count()
        self.mesh = Mesh(
            mesh_utils.create_device_mesh((jax.device_count() // local_cnt, local_cnt)),
            axis_names=("replica", "data"),
        )

    def curr_dist(self, itr):
        # dL_k/dn = alpha_k * (L_k - irreducible_k) / n_k, normalized over k
        if itr < self.start_step or self.current_params is None:
            return None

        irred_loss_K = np.exp(self.current_params[:, 1])
        expnt_K = np.maximum(self.current_params[:, 2], 0.05)  # bound the exponent away from 0
        curr_loss_K = scaling_law(
            (self.current_params[:, 0], self.current_params[:, 1], self.current_params[:, 2]),
            self.task_cnt_TxK.sum(0),
        )
        n_seen_K = np.float32(np.sum(self.task_cnt_TxK, 0))

        slope_K = expnt_K * (curr_loss_K - irred_loss_K)
        if self.divide_by_n:
            slope_K = slope_K / n_seen_K
        self.p_scaling = slope_K / slope_K.sum()

        weighted_slope_K = slope_K * self.p_history**0.5
        if self.use_prior:
            weighted_slope_K = weighted_slope_K * self.empirical_dist

        dist_K = clip_min_probability(weighted_slope_K / weighted_slope_K.sum(), 1e-2)
        self.p = 0.9 * self.p_avg + 0.1 * dist_K
        self.p_avg = (1 - 1 / (itr + 1)) * self.p_avg + 1 / (itr + 1) * dist_K
        return self.p

    def update(self, task_ids, losses, itr, lr, metrics=None, **kwargs):
        counts_K = np.zeros(self.k, dtype=np.int32)
        unique_values, counts = np.unique(task_ids, return_counts=True)
        counts_K[unique_values] = counts

        self.tokens_remaining = np.clip(
            self.tokens_remaining - counts_K * self.block_size, 1, self.tokens_remaining.max()
        )
        self.remain_dist = self.tokens_remaining / self.tokens_remaining.sum()

        increment = np.ones_like(counts_K[None]) * counts_K.sum() if self.use_same_step_size else counts_K[None]
        self.task_cnt_TxK = np.append(self.task_cnt_TxK, increment, axis=0)

        gamma = 0.9
        count_dist = counts_K / counts_K.sum()
        self.p_history = gamma * self.p_history + (1 - gamma) * count_dist
        self.current_lr = lr

        avg_loss = np.mean(losses).item()
        task_loss_K = np.zeros((self.k,))
        for i in range(self.k):
            if np.sum(task_ids == i) > 0:
                task_loss_K[i] = np.mean(losses[np.where(task_ids == i)])
            elif self.task_loss_TxK.shape[0] > 0:
                task_loss_K[i] = self.task_loss_TxK[-1, i]
            else:
                task_loss_K[i] = avg_loss
        self.task_loss_TxK = np.append(self.task_loss_TxK, task_loss_K[None], axis=0)

        if metrics is not None:
            for i in range(self.k):
                metrics[f"count_dist/task{i}"] = count_dist[i]
                metrics[f"p_history/task{i}"] = self.p_history[i]

        if itr >= self.start_step and itr % self.update_interval == 0:
            start_time = time.time()
            print(f"Refitting parameters at step {itr}...")
            current_params = self.compute_scaling_law_params()
            print(f"Took {time.time() - start_time:.1f}s.")
            if self.check_valid(current_params):
                self.current_params = current_params

        if metrics is not None and self.current_params is not None:
            self._log_param_metrics(metrics)

    def _log_param_metrics(self, metrics):
        for i in range(self.k):
            metrics[f"param_mult/task{i}"] = np.exp(self.current_params[i, 0])
            metrics[f"param_irred/task{i}"] = np.exp(self.current_params[i, 1])
            metrics[f"param_expon/task{i}"] = self.current_params[i, 2]
            metrics[f"remaining_dist/task{i}"] = self.remain_dist[i]

    def compute_epiplexity(self):
        if self.task_loss_TxK.shape[0] == 0:
            return {}

        counts_K = self.task_cnt_TxK[-1, :].astype(np.float64)
        curr_train_loss = self.task_loss_TxK[-1, :]
        self.train_epi_ceiling += counts_K * curr_train_loss
        self.train_total_nk += counts_K
        train_epi_K = self.train_epi_ceiling - curr_train_loss * self.train_total_nk

        metrics = {f"train_epi/task{i}": float(train_epi_K[i]) for i in range(self.k)}
        metrics["train_epi/total"] = float(train_epi_K.sum())

        if self.val_loss_SxK.shape[0] > 0:
            val_epi_K = self.val_epi_ceiling - self.val_loss_SxK[-1, :] * self.val_total_nk
            metrics.update({f"val_epi/task{i}": float(val_epi_K[i]) for i in range(self.k)})
            metrics["val_epi/total"] = float(val_epi_K.sum())
        return metrics

    def update_val_loss(self, loss_np_N, taskid_np_N, step: int):
        val_loss_K = np.array([np.mean(loss_np_N[taskid_np_N == i]) for i in range(self.k)])
        self.val_loss_SxK = np.append(self.val_loss_SxK, val_loss_K[None], axis=0)

        last_step = self.val_step_S[-1] if len(self.val_step_S) > 0 else 0
        interval_cnt_K = self.task_cnt_TxK[last_step:step].sum(0).astype(np.float64)
        self.val_cnt_SxK = np.append(self.val_cnt_SxK, interval_cnt_K[None], axis=0)
        self.val_epi_ceiling += interval_cnt_K * val_loss_K
        self.val_total_nk += interval_cnt_K
        self.val_step_S = np.append(self.val_step_S, step)

    def state_dict(self):
        # everything but the mesh; jax arrays become numpy so the dict pickles on any host
        to_np = lambda x: np.asarray(x) if isinstance(x, jax.Array) else x  # noqa: E731
        return {k: jax.tree_util.tree_map(to_np, v) for k, v in self.__dict__.items() if k != "mesh"}

    def load_state_dict(self, state):
        self.__dict__.update(state)

    def check_valid(self, params):
        nans, infs = np.isnan(np.sum(params)).item(), np.isinf(np.sum(params)).item()
        if nans or infs:
            print(f"Discarding fit: nans={nans}, infs={infs}.")
            return False
        return True

    def _make_global_array(self, arr, sharding):
        arrays = [
            jax.device_put(arr[index], d) for d, index in sharding.addressable_devices_indices_map(arr.shape).items()
        ]
        return jax.make_array_from_single_device_arrays(arr.shape, sharding, arrays)

    def compute_scaling_law_params(self):
        cum_cnt_TxK = np.cumsum(self.task_cnt_TxK, 0)
        if self.y_axis == "train_loss":
            cum_cnt_KxT = jnp.asarray(cum_cnt_TxK[self.ignore_steps :: 10].T)
            # Per-step training loss is noisy; smooth before fitting
            task_loss_KxT = jnp.asarray(
                savgol_filter(self.task_loss_TxK.T, 101, 3, axis=-1)[:, self.ignore_steps :: 10]
            )
        else:
            mask = self.val_step_S >= self.ignore_steps
            cum_cnt_KxT = jnp.asarray(cum_cnt_TxK[self.val_step_S[mask], :].T)
            task_loss_KxT = jnp.asarray(self.val_loss_SxK[mask].T)
        cum_cnt_KxT, task_loss_KxT = multihost_utils.broadcast_one_to_all((cum_cnt_KxT, task_loss_KxT))

        sharding = NamedSharding(self.mesh, P())
        cum_cnt_KxT = self._make_global_array(cum_cnt_KxT, sharding)
        task_loss_KxT = self._make_global_array(task_loss_KxT, sharding)

        fit_fn = functools.partial(fit_scaling_law, mesh=self.mesh)
        params_Kx3, _ = jax.vmap(fit_fn, in_axes=(0, 0))(cum_cnt_KxT, task_loss_KxT)
        return multihost_utils.process_allgather(params_Kx3, tiled=True)


def clip_min_probability(probs, min_prob):
    # rescale probs so every entry is at least min_prob and the total stays 1.
    total_deficit = max(min_prob * len(probs) - probs.sum(), 0)
    scaled_probs = probs * (1 - total_deficit) / probs.sum()
    clipped_probs = np.maximum(scaled_probs, min_prob)
    return clipped_probs / clipped_probs.sum()


def huber_loss(target, pred, delta=1e-3):
    abs_diff = jnp.abs(target - pred)
    return jnp.where(abs_diff > delta, delta * (abs_diff - 0.5 * delta), 0.5 * abs_diff**2)


def logsumexp(a, axis=None):
    a_max = jnp.max(a, axis=axis, keepdims=True)
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return jnp.squeeze(log_sum_exp, axis=axis) if axis is not None else log_sum_exp


def loss_fn(params_3, D, L):
    # huber loss in log space, plus barrier terms keeping the fit in a sane region
    a, e, alpha = params_3
    arg1 = a - alpha * jnp.log(D)
    arg2 = e + jnp.zeros(arg1.shape)
    pred = logsumexp(jnp.stack((arg1, arg2), axis=0), axis=0)

    barriers = (
        jnp.maximum(alpha - 0.8, 0)  # alpha <= 0.8
        - jnp.minimum(alpha, 0.001)  # alpha >= 0
        + jnp.maximum(a - 6.5, 0)  # a <= 6.5
        - jnp.minimum(e - 0.5, 0)  # e >= 0.5
    )
    return jnp.sum(huber_loss(jnp.log(L), pred)) + barriers


def scaling_law(params, D):
    a, e, alpha = params
    return np.exp(e) + np.exp(a) / D**alpha


@functools.partial(jax.jit, static_argnums=(2,))
def fit_scaling_law(D_T, losses_T, mesh):
    # LBFGS from a grid of inits; returns the params of the best-converged run
    init_a = jnp.arange(-2, 6, 1)
    init_e = jnp.arange(-2, 2, 0.5)
    init_alpha = jnp.arange(0, 0.8, 0.1)

    X, Y, Z = jnp.meshgrid(init_a, init_e, init_alpha, indexing="ij")
    init_grid_Gx3 = jnp.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    init_grid_Gx3 = jax.lax.with_sharding_constraint(
        init_grid_Gx3, NamedSharding(mesh, P(("replica", "data"), None))
    )

    solver = LBFGS(loss_fn, tol=1e-5, jit=True, maxiter=200)
    result_Gx3, state = jax.vmap(solver.run, in_axes=(0, None, None))(init_grid_Gx3, D_T, losses_T)
    return result_Gx3[jnp.nanargmin(state.value)], state

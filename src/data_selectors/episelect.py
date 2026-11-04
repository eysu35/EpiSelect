import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P
from scipy.signal import savgol_filter

from src.data_selectors.ado import AdoSelector, clip_min_probability
from src.scaling_law import epiplexity_grads, fit_scaling_law, readout_params


class EpiSelect(AdoSelector):
    def __init__(
        self,
        use_prior,
        use_credit_assignment,
        do_global_avg,
        ema,
        fit_freq,
        window_length,
        polyorder,
        tau,
        diag_gamma=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diag_gamma = diag_gamma
        self.use_prior = use_prior
        self.use_credit_assignment = use_credit_assignment
        self.do_global_avg = do_global_avg
        self.ema = ema
        self.fit_freq = fit_freq
        self.window_length = window_length
        self.polyorder = polyorder
        self.tau = tau

        self.fit_val = None
        self.fit_iters = None

    def curr_dist(self, itr):
        if itr < self.start_step or self.current_params is None:
            return None

        D = jnp.sum(self.task_cnt_TxK, axis=0, keepdims=True).T  # [K,1]
        weighted_slope_K = self._get_S_grads(D=D, current_params=self.current_params)  # [K,]
        self.p_scaling = weighted_slope_K / weighted_slope_K.sum()

        if self.use_prior:
            weighted_slope_K = weighted_slope_K * self.empirical_dist
        if self.use_credit_assignment:
            weighted_slope_K = weighted_slope_K * self.p_history**0.5

        dist_K = weighted_slope_K / weighted_slope_K.sum()

        self.p = clip_min_probability(self.ema * self.p_avg + (1.0 - self.ema) * dist_K, 1e-8)
        if self.do_global_avg:
            self.p_avg = (1 - 1 / (itr + 1)) * self.p_avg + (1 / (itr + 1)) * self.p
        else:
            self.p_avg = self.p
        return self.p

    def _log_param_metrics(self, metrics):
        K = self.k
        metrics["fit_val"] = float(self.fit_val)
        metrics["fit_iters"] = float(self.fit_iters)
        gamma_KxK, alpha_K, beta_K, irred_K = readout_params(self.current_params, K, self.diag_gamma)
        for i in range(K):
            metrics[f"param_alpha/task{i}"] = float(alpha_K[i])
            metrics[f"param_beta/task{i}"] = float(beta_K[i])
            metrics[f"param_irred/task{i}"] = float(irred_K[i])
            metrics[f"remaining_dist/task{i}"] = self.remain_dist[i]
            for j in range(K):
                metrics[f"param_gamma/task{i}_from{j}"] = float(gamma_KxK[i, j])

    def compute_scaling_law_params(self):
        cum_cnt_TxK = np.cumsum(self.task_cnt_TxK, axis=0)
        if self.y_axis == "train_loss":
            cum_cnt_KxT = jnp.asarray(cum_cnt_TxK[self.ignore_steps :: self.fit_freq].T)
            # per-step training loss is noisy; smooth before fitting
            task_loss_KxT = savgol_filter(self.task_loss_TxK.T, self.window_length, self.polyorder, axis=-1)
            task_loss_KxT = jnp.asarray(task_loss_KxT[:, self.ignore_steps :: self.fit_freq])
        else:
            mask = self.val_step_S >= self.ignore_steps
            cum_cnt_KxT = jnp.asarray(cum_cnt_TxK[self.val_step_S[mask], :].T)
            task_loss_KxT = jnp.asarray(self.val_loss_SxK[mask].T)
        cum_cnt_KxT, task_loss_KxT = multihost_utils.broadcast_one_to_all((cum_cnt_KxT, task_loss_KxT))

        sharding = jax.sharding.NamedSharding(self.mesh, P())
        cum_cnt_KxT = self._make_global_array(cum_cnt_KxT, sharding)
        task_loss_KxT = self._make_global_array(task_loss_KxT, sharding)

        params_Kx3pK, state = self._fit_scaling_law_fn()(cum_cnt_KxT, task_loss_KxT)
        self.fit_val, self.fit_iters = state[0], state[1]
        return multihost_utils.process_allgather(params_Kx3pK, tiled=True)

    def _fit_scaling_law_fn(self):
        return functools.partial(fit_scaling_law, mesh=self.mesh, K=self.k, diag_gamma=self.diag_gamma)

    def _get_S_grads(self, D, current_params):
        return jnp.exp(epiplexity_grads(D, current_params, self.diag_gamma) / self.tau)

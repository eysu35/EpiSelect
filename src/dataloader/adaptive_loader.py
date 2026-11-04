import typing as tp
from abc import ABC, abstractmethod
from collections import Counter

import hydra
import jax
import numpy as np
from jax.experimental import multihost_utils

from src.data_selectors.ado import AdoSelector
from src.data_selectors.episelect import EpiSelect
from src.dataloader.abstract import Array
from src.sharding import get_data_sharding, get_shard_fn

from .abstract import Loader

SELECTORS = {
    "ado": AdoSelector,
    "episelect": EpiSelect,
}


class TasksSampler(ABC):
    @abstractmethod
    def __init__(self, split, g_accum_iters, block_size, **kwargs):
        pass

    @abstractmethod
    def get_batch(self, num_per_task, B) -> tp.Tuple[Array, Array, Array]:
        pass

    @property
    @abstractmethod
    def k(self) -> int:
        pass

    @property
    @abstractmethod
    def init_dist(self) -> np.ndarray:
        pass


class AdaptiveLoader(Loader):
    def __init__(
        self,
        batch_size: int,
        split: str,
        mesh,
        block_size: int,
        g_accum_iters: int,
        data_dir: str,
        tasks_sampler_cfg,
        update_interval: int,
        ignore_steps: int,  # skip this many initial steps when fitting
        start_step: int,  # first step that uses the fitted distribution
        use_prior: bool,
        use_credit_assignment: bool,
        do_global_avg: bool,
        ema: float,
        fit_freq: int,
        window_length: int,
        polyorder: int,
        tau: float,
        mode: str = "empirical",  # empirical | uniform
        diag_gamma: bool = False,  # fix gamma to the identity (per-domain law)
        policy_type: str = "episelect",
        y_axis: str = "train_loss",  # train_loss | val_loss
        use_same_step_size: bool = False,
        seed: int = 0,
        **kwargs,
    ):
        self.n_procs = jax.process_count()
        self.rng = np.random.default_rng(seed + jax.process_index())
        self.sharding_fn = get_shard_fn(mesh, get_data_sharding(mesh))
        self.block_size = block_size
        self.g_accum_iters = g_accum_iters
        self.batch_size = batch_size

        self.tasks_sampler: TasksSampler = hydra.utils.instantiate(tasks_sampler_cfg, split, block_size, data_dir, seed)
        self.k = self.tasks_sampler.k
        task_token_count = self.tasks_sampler.total_token_n if mode == "empirical" else np.ones(self.k)

        selector_kwargs = dict(
            n_task=self.k,
            task_token_count=task_token_count,
            block_size=block_size,
            ignore_steps=ignore_steps,
            start_step=start_step,
            y_axis=y_axis,
            update_interval=update_interval,
            use_same_step_size=use_same_step_size,
            use_prior=use_prior,
        )
        if policy_type != "ado":
            selector_kwargs.update(
                use_credit_assignment=use_credit_assignment,
                do_global_avg=do_global_avg,
                ema=ema,
                fit_freq=fit_freq,
                window_length=window_length,
                polyorder=polyorder,
                tau=tau,
                diag_gamma=diag_gamma,
            )
        if policy_type not in SELECTORS:
            raise ValueError(f"Unrecognized policy_type: {policy_type}. Options: {sorted(SELECTORS)}")
        self.data_policy = SELECTORS[policy_type](**selector_kwargs)

        self._curr_dist0 = None  # host 0's distribution, broadcast to all hosts
        self.cumulative_cnts = Counter()
        print(f"Init dist: {self.curr_dist0}.")

    @property
    def curr_dist0(self):
        if self._curr_dist0 is None:
            return self.data_policy.empirical_dist
        return self._curr_dist0

    def get_batch(self, itr: int):
        B = self.batch_size // self.n_procs
        BG = B * self.g_accum_iters

        task_probs_K = np.float64(self.curr_dist0)
        task_probs_K /= task_probs_K.sum()
        idx = self.rng.choice(np.arange(self.k), BG, p=task_probs_K, replace=True)
        cnt = Counter(idx)
        num_per_task = np.array([cnt[i] for i in range(self.k)])

        x_BGxT, y_BGxT, task_ids_BG = self.tasks_sampler.get_batch(num_per_task, BG)
        x_GxBxT = x_BGxT.reshape(self.g_accum_iters, B, self.block_size)
        y_GxBxT = y_BGxT.reshape(self.g_accum_iters, B, self.block_size)
        task_ids_GxB = task_ids_BG.reshape(self.g_accum_iters, B)
        info = {"task_ids": task_ids_GxB[..., None], "p": task_probs_K[task_ids_GxB][..., None]}
        return jax.tree_util.tree_map(self.sharding_fn, (x_GxBxT, y_GxBxT, info))

    def update_val_loss(self, loss_np_N, taskid_np_N, step: int):
        self.data_policy.update_val_loss(loss_np_N, taskid_np_N, step)

    def state_dict(self):
        return {
            "host": {"rng": self.rng.bit_generator.state, "consumed": self.tasks_sampler.consumed.tolist()},
            "global": {
                "policy": self.data_policy.state_dict(),
                "cumulative_cnts": dict(self.cumulative_cnts),
                "curr_dist0": self._curr_dist0,
            },
        }

    def load_state_dict(self, host, glob):
        self.rng.bit_generator.state = host["rng"]
        self.tasks_sampler.skip(host["consumed"])
        self.data_policy.load_state_dict(glob["policy"])
        self.cumulative_cnts = Counter(glob["cumulative_cnts"])
        self._curr_dist0 = glob["curr_dist0"]

    def update(self, loss_GxB: jax.Array, info: tp.Dict, itr: int, lr: float):
        metrics = {}
        loss_GxB, info = multihost_utils.process_allgather((loss_GxB, info), tiled=True)
        task_id_GxB = info["task_ids"].squeeze(-1)
        self.cumulative_cnts.update(task_id_GxB.reshape(-1).tolist())
        self.data_policy.update(task_id_GxB, loss_GxB, itr, lr=lr, metrics=metrics)

        if self.data_policy.task_loss_TxK.shape[0] > 0:
            last_losses = self.data_policy.task_loss_TxK[-1]
            for i in range(self.k):
                metrics[f"loss/task{i}"] = float(last_losses[i])

        for i in range(self.k):
            metrics[f"cum_task_cnts/{i}"] = self.cumulative_cnts[i]
            metrics[f"task_prob/task{i}"] = self.curr_dist0[i]
            metrics[f"p_scaling/task{i}"] = self.data_policy.p_scaling[i]
            metrics[f"tokens/domain{i}"] = self.cumulative_cnts[i] * self.block_size
        metrics["tokens/total"] = sum(self.cumulative_cnts.values()) * self.block_size
        metrics.update(self.data_policy.compute_epiplexity())

        curr_dist = self.data_policy.curr_dist(itr)
        if curr_dist is None:
            return metrics

        if self.n_procs > 1:
            self._curr_dist0 = multihost_utils.process_allgather(curr_dist, tiled=False)[0]
        else:
            self._curr_dist0 = curr_dist
        return metrics

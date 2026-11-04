from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, vmap
from jax.nn import logsumexp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxopt import LBFGS


def huber_loss(y_hat, y, delta=1e-3):
    abs_diff = jnp.abs(y - y_hat)
    part1 = delta * (abs_diff - 0.5 * delta)
    part2 = 0.5 * abs_diff**2
    return jnp.where(abs_diff > delta, part1, part2)


def sample_tilted_simplex(key, K, num_samples, tilt):
    # dirichlet init
    keys = jr.split(key, K)
    all_gammas = []
    for i in range(K):
        alpha = jnp.ones(K).at[i].set(tilt)
        all_gammas.append(jr.dirichlet(keys[i], alpha, shape=(num_samples,)))
    return jnp.stack(all_gammas, axis=0)  # [K,G,K]


def construct_sparse_inits(key, K):
    # grid over the scalars, crossed with one tilted-simplex draw for gamma:
    #   alpha in {0.0, 0.1, ..., 0.8}, log_beta in {-2, -1, ..., 6}, log_eps in {-2.0, -1.5, ..., 2.0}
    # 9*9*9 = 729 inits. arange excludes its stop, hence the extra step on each bound
    log_beta = jnp.arange(-2, 7, 1)
    log_eps = jnp.arange(-2, 2.5, 0.5)
    alpha = jnp.arange(0, 0.9, 0.1)
    gamma_grid_n = 1
    gamma = sample_tilted_simplex(key, K, num_samples=gamma_grid_n, tilt=10.0)  # [K,G,K]

    A, B, E = jnp.meshgrid(alpha, log_beta, log_eps, indexing="ij")
    scalars = jnp.stack([A, B, E], axis=-1).reshape(-1, 3)  # [S,3]
    scalars = jnp.broadcast_to(scalars[None], shape=(K,) + scalars.shape)  # [K,S,3]

    scalars_tiled = jnp.tile(scalars, (1, gamma.shape[1], 1))  # [K,GS,3]
    gamma_expanded = jnp.repeat(gamma, scalars.shape[1], axis=-2)  # [K,GS,K]
    gamma_expanded = jnp.transpose(gamma_expanded, axes=[2, 1, 0])

    init_grid = jnp.concatenate([jnp.log(gamma_expanded), scalars_tiled], axis=-1)  # [K,GS,K+3]
    init_grid = init_grid.transpose(1, 2, 0)  # [GS,K+3,K]
    return init_grid.reshape(init_grid.shape[0], -1)  # [GS,K(K+3)]


def _unpack(theta, K, diag_gamma=False):
    K2 = K * K
    if diag_gamma:  # per-domain law: gamma fixed to the identity, its entries in theta are unused
        log_gamma = jnp.where(jnp.eye(K, dtype=bool), 0.0, -jnp.inf)
    else:
        log_gamma = theta[:K2].reshape(K, K)
        log_gamma = log_gamma - logsumexp(log_gamma, axis=-1, keepdims=True)
    return log_gamma, theta[K2 : K2 + K], theta[K2 + K : K2 + 2 * K], theta[-K:]


def _log_loss(theta, x, diag_gamma=False):
    # theta: [K(K+3),], x: [T,K] -> [T,K]
    log_gamma, alpha, log_beta, log_eps = _unpack(theta, K=x.shape[-1], diag_gamma=diag_gamma)
    term1 = logsumexp(log_gamma[None] + jnp.log(x)[:, None], axis=-1)  # [T,K]
    term1 = log_beta[None] - alpha[None] * term1
    term2 = log_eps[None] * jnp.ones_like(x)
    return logsumexp(jnp.stack([term1, term2], axis=0), axis=0)


def _fit_objective(params, D, L, diag_gamma=False):
    # D, L: [K,T]. huber on log-loss, plus barriers keeping the fit in a sane region
    log_L_hat = _log_loss(params, jnp.transpose(D, axes=[1, 0]), diag_gamma)  # [T,K]
    log_L = jnp.log(jnp.transpose(L, axes=[1, 0]))  # [T,K]
    _, alpha, log_beta, log_eps = _unpack(params, D.shape[0], diag_gamma)

    alpha_max = jnp.sum(jnp.maximum(alpha - 0.8, 0))  # alpha <= 0.8
    alpha_min = jnp.sum(-jnp.minimum(alpha, 0.001))  # alpha >= 0
    log_beta_max = jnp.sum(jnp.maximum(log_beta - 6.5, 0))  # beta <= ~665
    log_eps_min = jnp.sum(-jnp.minimum(log_eps - 0.5, 0))  # eps >= 0.5

    out = jnp.sum(huber_loss(log_L, log_L_hat, delta=1e-3))
    return out + alpha_max + alpha_min + log_beta_max + log_eps_min


def readout_params(theta, K, diag_gamma=False):
    log_gamma, alpha, log_beta, log_eps = _unpack(theta, K=K, diag_gamma=diag_gamma)
    return np.exp(log_gamma), np.array(alpha), np.exp(log_beta), np.exp(log_eps)


def epiplexity_grads(tokens, params, diag_gamma=False):
    log_gamma, alpha, _, log_eps = _unpack(params, K=tokens.shape[0], diag_gamma=diag_gamma)
    gamma = np.exp(log_gamma)  # [K,K]
    alpha = np.array(alpha)  # [K,]
    eps = np.exp(log_eps)  # [K,]
    loss = np.exp(_log_loss(params, tokens.T, diag_gamma).T)[:, 0]  # [K,]

    dS = (alpha * (loss - eps))[:, None]  # [K,1]
    num = gamma * tokens  # [K,K]
    denom = np.sum(gamma * tokens.T, axis=-1, keepdims=True)  # [K,1]
    return np.sum((num / denom) * dS, axis=0)  # [K,]


@partial(jit, static_argnums=(2, 3, 4))
def fit_scaling_law(D, L, mesh, K, diag_gamma=False):
    init_grid = construct_sparse_inits(key=jr.PRNGKey(21), K=K)  # [G,K(K+3)]
    sharding = NamedSharding(mesh, P(("replica", "data"), None))
    init_grid = jax.lax.with_sharding_constraint(init_grid, sharding)

    solver = LBFGS(partial(_fit_objective, diag_gamma=diag_gamma), tol=1e-5, jit=True, maxiter=200)
    all_theta_hats, state = vmap(solver.run, in_axes=(0, None, None))(init_grid, D, L)
    min_pos = jnp.nanargmin(state.value)
    return all_theta_hats[min_pos], (state.value[min_pos], state.iter_num[min_pos])

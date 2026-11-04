import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.sharding import Mesh

from src.scaling_law import construct_sparse_inits, epiplexity_grads, fit_scaling_law, readout_params


def np_loss(gamma, alpha, beta, eps, x):
    # reference implementation of the cross-domain law, in plain loops
    # gamma: [K,K], alpha/beta/eps: [K,], x: [N,K]
    preds = np.zeros(shape=x.shape, dtype=x.dtype)
    for i in range(x.shape[0]):
        for m in range(x.shape[1]):
            for k in range(x.shape[1]):
                preds[i, m] += gamma[m, k] * x[i, k]
            preds[i, m] = beta[m] / preds[i, m] ** alpha[m] + eps[m]
    return preds


def np_grads(gamma, alpha, beta, eps, x):
    # reference dS/dn_k. x: [1,K]
    assert x.shape[0] == 1
    loss = np_loss(gamma=gamma, alpha=alpha, beta=beta, eps=eps, x=x)

    denom = np.zeros(shape=x.shape[-1], dtype=x.dtype)
    for m in range(x.shape[1]):
        denom[m] = np.sum(gamma[m, :] * x[0, :])

    grads = np.zeros(shape=x.shape[-1], dtype=x.dtype)
    for j in range(x.shape[1]):
        for m in range(x.shape[1]):
            grads[j] += x[0, m] * alpha[m] * (gamma[m, j] / denom[m]) * (loss[0, m] - eps[m])
    return grads


def test_fit_scaling_law():
    devices = np.array(jax.devices())
    mesh = Mesh(devices.reshape(1, len(devices)), ("replica", "data"))
    T, K, dtype, key = 111, 17, jnp.float32, jr.PRNGKey(18)
    D = jr.randint(key, minval=1, maxval=100, shape=(K, T)).astype(dtype)
    L = jr.normal(key, shape=(K, T), dtype=dtype) + 1.8
    theta_hat, _ = fit_scaling_law(D, L, mesh, K=K)
    assert theta_hat.shape == (K * (K + 3),)


def test_epiplexity_grads_match_reference():
    dtype = jnp.float32
    N = 10
    x1, x2 = jnp.linspace(10, 50, N), jnp.linspace(30, 70, N)
    X, Y = jnp.meshgrid(x1, x2, indexing="ij")
    x_grid = jnp.stack((X.flatten(), Y.flatten()), axis=1)
    K = x_grid.shape[-1]

    log_gamma = jnp.array([[-0.1, -2.3], [-1.6, -0.2]], dtype=dtype)
    alpha = jnp.array([0.6, 0.3], dtype=dtype)
    log_beta = jnp.array([0.0, -0.5], dtype=dtype)
    log_eps = jnp.array([0.6, 0.7], dtype=dtype)
    theta = jnp.concat([log_gamma.flatten(), alpha.flatten(), log_beta.flatten(), log_eps])
    tokens = jnp.sum(x_grid, axis=0, keepdims=True)

    approx = epiplexity_grads(tokens.T, params=theta)
    assert approx.shape == (K,)

    # the fit normalizes each gamma row onto the simplex, so the reference must too
    gamma = np.exp(log_gamma) / np.sum(np.exp(log_gamma), axis=-1, keepdims=True)
    soln = np_grads(gamma, alpha, np.exp(log_beta), np.exp(log_eps), x=tokens)
    assert np.sum(np.abs(soln - approx)) < 1e-7


def test_construct_sparse_inits_shape():
    # 9 alpha x 9 log_beta x 9 log_eps = 729, tiled by gamma_grid_n = 1
    K = 2
    assert construct_sparse_inits(jr.PRNGKey(21), K).shape == (729, K * (K + 3))


def test_construct_sparse_inits_layout():
    K = 3
    K2 = K * K
    theta = np.array(construct_sparse_inits(key=jr.PRNGKey(0), K=K)[0])  # [K(K+3)]

    log_gamma = theta[:K2].reshape(K, K)
    alpha, log_beta, log_eps = theta[K2 : K2 + K], theta[K2 + K : K2 + 2 * K], theta[-K:]

    # each scalar is broadcast to K identical values
    assert np.allclose(alpha, alpha[0])
    assert np.allclose(log_beta, log_beta[0])
    assert np.allclose(log_eps, log_eps[0])

    gamma = np.exp(log_gamma)
    assert all(gamma[m, m] == max(gamma[m]) for m in range(K)), "inits should be diagonally dominant"
    assert not all(np.allclose(gamma[0], gamma[m]) for m in range(1, K)), "rows should differ"
    assert all(np.allclose(1.0, gamma[m].sum()) for m in range(K)), "rows should lie on the simplex"


def test_diag_gamma_fit_fixes_gamma_to_identity():
    devices = np.array(jax.devices())
    mesh = Mesh(devices.reshape(1, len(devices)), ("replica", "data"))
    T, K, key = 60, 4, jr.PRNGKey(3)
    D = jnp.cumsum(jr.randint(key, minval=1, maxval=100, shape=(K, T)), axis=1).astype(jnp.float32)
    L = 2.0 + 3.0 * D**-0.3 + 0.01 * jr.normal(key, shape=(K, T), dtype=jnp.float32)  # positive, decreasing
    theta_hat, _ = fit_scaling_law(D, L, mesh, K=K, diag_gamma=True)
    assert theta_hat.shape == (K * (K + 3),)
    gamma, alpha, beta, eps = readout_params(theta_hat, K, diag_gamma=True)
    assert np.array_equal(gamma, np.eye(K))
    assert np.all(np.isfinite(alpha)) and np.all(np.isfinite(beta)) and np.all(np.isfinite(eps))


def test_epiplexity_grads_diag_gamma_is_per_domain():
    dtype = jnp.float32
    log_gamma = jnp.array([[-0.1, -2.3], [-1.6, -0.2]], dtype=dtype)  # ignored under diag_gamma
    alpha = jnp.array([0.6, 0.3], dtype=dtype)
    log_beta = jnp.array([0.0, -0.5], dtype=dtype)
    log_eps = jnp.array([0.6, 0.7], dtype=dtype)
    theta = jnp.concat([log_gamma.flatten(), alpha.flatten(), log_beta.flatten(), log_eps])
    tokens = jnp.array([[300.0, 500.0]], dtype=dtype)

    approx = epiplexity_grads(tokens.T, params=theta, diag_gamma=True)
    soln = np_grads(np.eye(2), alpha, np.exp(log_beta), np.exp(log_eps), x=tokens)
    assert np.sum(np.abs(soln - approx)) < 1e-6
    # differs from the cross-domain law with the same theta
    assert np.sum(np.abs(epiplexity_grads(tokens.T, params=theta) - approx)) > 1e-3

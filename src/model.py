import math
import typing as tp

import equinox as eqx
import jax

from .layers import (
    Embedding,
    Linear,
    RMSNorm,
    apply_rotary_pos_emb,
    fixed_pos_embedding,
)

jnp, jrandom, vmap, jtu = jax.numpy, jax.random, jax.vmap, jax.tree_util
Array = jax.Array
KeyArray = tp.Any
P = jax.sharding.PartitionSpec
NamedSharding = jax.sharding.NamedSharding
Mesh = jax.sharding.Mesh
with_sharding_constraint = jax.lax.with_sharding_constraint


class SwiGLU(eqx.Module):
    w1: Linear
    w2: Linear
    w3: Linear
    dropout: eqx.nn.Dropout

    def __init__(self, n_embd, dropout, key):
        key1, key2, key3 = jrandom.split(key, 3)
        hidden_dim = int(2 * (4 * n_embd) / 3)  # adjust to match pre-SwiGLU transformer size
        multiple_of = 8  # hidden dim must be a multiple of this for sharding
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = Linear(n_embd, hidden_dim, key=key1)
        self.w2 = Linear(hidden_dim, n_embd, key=key2)
        self.w3 = Linear(n_embd, hidden_dim, key=key3)
        self.dropout = eqx.nn.Dropout(dropout)

    @jax.named_scope("mlp")
    def __call__(self, x_D, inference=False, key=None):
        x_D = self.w2(jax.nn.silu(self.w1(x_D)) * self.w3(x_D))
        return self.dropout(x_D, inference=inference, key=key)


class CausalSelfAttention(eqx.Module):
    n_head: int
    n_embd: int
    c_attn: Linear
    c_proj: Linear
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout
    q_ln: tp.Optional[eqx.nn.LayerNorm]
    k_ln: tp.Optional[eqx.nn.LayerNorm]

    def __init__(self, n_embd, n_head, dropout, key, mup=False):
        key1, key2 = jrandom.split(key)
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.c_attn = Linear(n_embd, 3 * n_embd, key=key1)
        self.c_proj = Linear(n_embd, n_embd, key=key2)
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.resid_dropout = eqx.nn.Dropout(dropout)
        self.q_ln, self.k_ln = None, None
        if not mup:
            self.q_ln = eqx.nn.LayerNorm(n_embd // n_head, eps=1e-6, use_weight=True, use_bias=False)
            self.k_ln = eqx.nn.LayerNorm(n_embd // n_head, eps=1e-6, use_weight=True, use_bias=False)

    @jax.named_scope("causal_sa")
    def __call__(self, x_TxD, inference=False, key=None):
        adrop_key, pdrop_key = jrandom.split(key) if key is not None else (None, None)
        T, D = x_TxD.shape
        Q_TxD, K_TxD, V_TxD = jnp.split(vmap(self.c_attn)(x_TxD), 3, axis=-1)
        C = self.n_embd // self.n_head
        Q_HxTxC = jnp.transpose(jnp.reshape(Q_TxD, (T, self.n_head, C)), (1, 0, 2))
        K_HxTxC = jnp.transpose(jnp.reshape(K_TxD, (T, self.n_head, C)), (1, 0, 2))
        if self.q_ln is not None:  # QK LayerNorm
            Q_HxTxC = vmap(vmap(self.q_ln))(Q_HxTxC)
            K_HxTxC = vmap(vmap(self.k_ln))(K_HxTxC)
        # Rotary embeddings
        sin_TxCp, cos_TxCp = fixed_pos_embedding(C, T)  # Cp = C//2
        Q_HxTxC = apply_rotary_pos_emb(Q_HxTxC, sin_TxCp, cos_TxCp)
        K_HxTxC = apply_rotary_pos_emb(K_HxTxC, sin_TxCp, cos_TxCp)
        V_HxTxC = jnp.transpose(jnp.reshape(V_TxD, (T, self.n_head, C)), (1, 0, 2))
        A_HxTxT = Q_HxTxC @ jnp.transpose(K_HxTxC, (0, 2, 1))
        causal_mask = jnp.tril(jnp.ones((1, T, T))) == 0
        A_HxTxT = jnp.where(causal_mask, float("-inf"), A_HxTxT)
        # Softmax should be in full precision
        orig_dtype = A_HxTxT.dtype
        # mup uses 1/C, while standard uses 1 / sqrt(C)
        scaling = math.sqrt(C) if self.q_ln is not None else C
        A_HxTxT = jax.nn.softmax(A_HxTxT.astype(jnp.float32) / scaling, axis=-1)
        A_HxTxT = A_HxTxT.astype(orig_dtype)
        A_HxTxT = self.attn_dropout(A_HxTxT, inference=inference, key=adrop_key)
        out_TxD = jnp.reshape(jnp.transpose(A_HxTxT @ V_HxTxC, (1, 0, 2)), (T, D))
        out_TxD = self.resid_dropout(vmap(self.c_proj)(out_TxD), inference=inference, key=pdrop_key)
        return out_TxD


class Block(eqx.Module):
    attn: CausalSelfAttention
    mlp: SwiGLU
    ln1: RMSNorm
    ln2: RMSNorm

    def __init__(self, n_embd, n_head, dropout, key, mup=False):
        key1, key2 = jrandom.split(key)
        self.attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout, key=key1, mup=mup)
        self.mlp = SwiGLU(n_embd=n_embd, dropout=dropout, key=key2)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    @jax.named_scope("block")
    def __call__(self, x_TxD, inference=False, key=None):
        attn_key, mlp_key = (None, None)
        if key is not None:
            attn_key, mlp_key = jrandom.split(key)
            mlp_key = jrandom.split(mlp_key, x_TxD.shape[0])
        x_TxD = x_TxD + self.attn(vmap(self.ln1)(x_TxD), inference=inference, key=attn_key)
        mlp = vmap(self.mlp, in_axes=(0, None, 0))
        return x_TxD + mlp(vmap(self.ln2)(x_TxD), inference, mlp_key)


class GPT(eqx.Module):
    wte: Embedding
    drop: eqx.nn.Dropout
    blocks: tp.List[Block]
    ln_f: RMSNorm
    lm_head: Linear
    n_layer: int

    def __init__(self, config, key):
        self.n_layer = config.n_layer
        block_key, embed_key, head_key = jrandom.split(key, 3)
        self.drop = eqx.nn.Dropout(config.dropout)

        def make_block(_key):
            return Block(config.n_embd, config.n_head, config.dropout, _key, mup=config.mup)

        self.blocks = eqx.filter_vmap(make_block)(jrandom.split(block_key, config.n_layer))
        self.ln_f = RMSNorm(config.n_embd, eps=1e-5)
        embed_std = 1 if config.mup else (1 / math.sqrt(config.n_embd))
        wte_wt = embed_std * jrandom.normal(embed_key, (config.vocab_size, config.n_embd))
        self.wte = Embedding(config.vocab_size, config.n_embd, weight=wte_wt)
        if config.mup:  # For our impl of mup, easier not to share
            self.lm_head = Linear(config.n_embd, config.vocab_size, key=head_key)
        else:  # Otherwise, share embed and final weights.
            self.lm_head = Linear(config.n_embd, config.vocab_size, weight=wte_wt)

    @jax.named_scope("gpt")
    def __call__(self, x_T, inference=False, key=None):
        # Either (inference=False and key) or (inference=True and key=None)
        drop_key, block_keys = None, None
        if key is not None:
            drop_key, block_keys = jrandom.split(key)
            block_keys = jrandom.split(block_keys, self.n_layer)
        x_TxD = self.drop(self.wte(x_T), inference=inference, key=drop_key)
        dynamic_blocks, static_blocks = eqx.partition(self.blocks, eqx.is_array)

        @jax.checkpoint
        def block_fn(_x_TxD: Array, block_and_key: tp.Tuple[GPT, tp.Optional[KeyArray]]):
            _dynamic_block, _key = block_and_key
            block = eqx.combine(_dynamic_block, static_blocks)
            return block(_x_TxD, inference=inference, key=_key), None

        # Set unroll=self.n_layer for better speed (but slower compile)
        x_TxD, _ = jax.lax.scan(block_fn, x_TxD, (dynamic_blocks, block_keys), unroll=1)
        x_TxD = vmap(self.ln_f)(x_TxD)
        logits_TxV = vmap(self.lm_head)(x_TxD)
        return logits_TxV


def _attn_decode_step(
    attn: CausalSelfAttention,
    x_D: Array,
    layer_kv: Array,
    pos: Array,
    sin_TxCp: Array,
    cos_TxCp: Array,
) -> tp.Tuple[Array, Array]:
    # single-token attention step using KV cache
    #
    # layer_kv: [2, max_len, n_head, head_dim] — K cache at [0], V cache at [1]
    # Returns: out_D [D], updated layer_kv
    C = attn.n_embd // attn.n_head
    qkv = attn.c_attn(x_D)  # [3*D]
    Q_HxC, K_HxC, V_HxC = [jnp.reshape(t, (attn.n_head, C)) for t in jnp.split(qkv, 3)]
    if attn.q_ln is not None:
        Q_HxC = vmap(attn.q_ln)(Q_HxC)
        K_HxC = vmap(attn.k_ln)(K_HxC)
    # RoPE at position pos — apply_rotary_pos_emb expects [H, T, C] + [T, C//2]
    Cp = sin_TxCp.shape[1]
    sin_1 = jax.lax.dynamic_slice(sin_TxCp, (pos, 0), (1, Cp))  # [1, C//2]
    cos_1 = jax.lax.dynamic_slice(cos_TxCp, (pos, 0), (1, Cp))  # [1, C//2]
    Q_HxC = apply_rotary_pos_emb(Q_HxC[:, None, :], sin_1, cos_1)[:, 0, :]
    K_HxC = apply_rotary_pos_emb(K_HxC[:, None, :], sin_1, cos_1)[:, 0, :]
    # Write to KV cache
    layer_kv = layer_kv.at[0, pos].set(K_HxC)
    layer_kv = layer_kv.at[1, pos].set(V_HxC)
    # Attend over all cached positions: Q [H, C] × K [H, C, max_len]
    K_HxCxT = jnp.transpose(layer_kv[0], (1, 2, 0))
    scaling = math.sqrt(C) if attn.q_ln is not None else C
    A_HxT = jnp.einsum("hc,hct->ht", Q_HxC, K_HxCxT) / scaling
    mask = jnp.arange(layer_kv.shape[1]) > pos
    A_HxT = jnp.where(mask[None], float("-inf"), A_HxT)
    A_HxT = jax.nn.softmax(A_HxT.astype(jnp.float32), axis=-1).astype(Q_HxC.dtype)
    out_HxC = jnp.einsum("ht,thc->hc", A_HxT, layer_kv[1])
    out_D = attn.c_proj(jnp.reshape(out_HxC, (attn.n_embd,)))
    return out_D, layer_kv


def _block_decode_step(
    block: Block,
    x_D: Array,
    layer_kv: Array,
    pos: Array,
    sin_TxCp: Array,
    cos_TxCp: Array,
) -> tp.Tuple[Array, Array]:
    h_D, new_kv = _attn_decode_step(block.attn, block.ln1(x_D), layer_kv, pos, sin_TxCp, cos_TxCp)
    x_D = x_D + h_D
    x_D = x_D + block.mlp(block.ln2(x_D), inference=True, key=None)
    return x_D, new_kv


def greedy_generate(model: "GPT", prompt: Array, max_new_tokens: int, eos_id: int) -> Array:
    # greedy decode with KV cache
    #
    # prompt: [T_p] int32
    # Returns: [T_p + max_new_tokens] int32 — prompt concatenated with generated tokens
    # (positions after EOS are filled with the token that follows EOS in greedy order)
    T_p = prompt.shape[0]
    n_head: int = model.blocks.attn.n_head
    D: int = model.blocks.attn.n_embd
    C = D // n_head
    max_len = T_p + max_new_tokens

    sin_TxCp, cos_TxCp = fixed_pos_embedding(C, max_len)

    dtype = jtu.tree_leaves(eqx.filter(model, eqx.is_array))[0].dtype
    kv = jnp.zeros((model.n_layer, 2, max_len, n_head, C), dtype=dtype)
    buf = jnp.zeros(max_len, dtype=jnp.int32).at[:T_p].set(prompt)

    dynamic_blocks, static_blocks = eqx.partition(model.blocks, eqx.is_array)

    def step(carry, t):
        kv, buf = carry
        x_D = model.wte(buf[t])  # [D]

        def layer_fn(x_D, layer_data):
            dynamic_block, layer_kv = layer_data
            block = eqx.combine(dynamic_block, static_blocks)
            x_D, new_layer_kv = _block_decode_step(block, x_D, layer_kv, t, sin_TxCp, cos_TxCp)
            return x_D, new_layer_kv

        # scan over layers: each layer gets its KV slice [2, max_len, n_head, C]
        # and returns the updated slice; stacked output reconstructs full kv
        x_D, new_kv = jax.lax.scan(layer_fn, x_D, (dynamic_blocks, kv))

        x_D = model.ln_f(x_D)
        next_tok = jnp.argmax(model.lm_head(x_D)).astype(jnp.int32)

        # Only write to buffer during decode phase (t >= T_p - 1 means we just
        # processed the last prompt token and are ready to emit the first new one)
        new_buf = jax.lax.cond(
            t >= T_p - 1,
            lambda: buf.at[t + 1].set(next_tok),
            lambda: buf,
        )
        return (new_kv, new_buf), None

    (_, buf), _ = jax.lax.scan(step, (kv, buf), jnp.arange(max_len - 1))
    return buf


def count_params(model: GPT) -> int:
    dupe = jnp.size(model.lm_head.weight_MxN)  # embedding and final layer are shared.
    tot = sum([jnp.size(x) for x in jtu.tree_leaves(model) if isinstance(x, jax.Array)])
    return tot - dupe  # non-embedding only.


def shard_gpt(model: GPT, mesh: Mesh, shard_model: bool, sharding_fn=with_sharding_constraint) -> eqx.Module:
    # shard model parameters over devices (TPUs or GPUs)

    def sharding_map(x: Array) -> NamedSharding:
        axes: tuple[tp.Any, ...] = (None,) * x.ndim
        if x.size > 2**18 and shard_model:
            axes = (None,) * (x.ndim - 1) + ("data",)
        return NamedSharding(mesh, P(*axes))

    dynamic_model, static_model = eqx.partition(model, eqx.is_array)
    dynamic_model = jtu.tree_map(lambda x: sharding_fn(x, sharding_map(x)), dynamic_model)
    return eqx.combine(dynamic_model, static_model)

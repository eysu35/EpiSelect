import typing as tp

import equinox as eqx
import jax
import lm_eval
import lm_eval.api.model
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from lm_eval.models.utils import Collator, handle_stop_sequences
from tqdm import tqdm
from transformers import AutoTokenizer

from src.harness.llm_harness_jax_utils import pad_and_concat
from src.model import greedy_generate

jnp, jrandom, vmap, scan, jtu = (
    jax.numpy,
    jax.random,
    jax.vmap,
    jax.lax.scan,
    jax.tree_util,
)


def cast_pytree(pytree: tp.Any, dtype: jnp.dtype) -> tp.Any:
    def cast(x):
        if eqx.is_array(x):
            return x.astype(dtype)
        return x

    return jtu.tree_map(cast, pytree)


def get_shard_fn(mesh, sharding):
    n_procs = jax.process_count()
    n_local_devices = len(mesh.local_devices)

    def shard(x):
        total_devices = n_procs * n_local_devices
        items_per_device = -(-x.shape[0] // total_devices)  # Ceiling division
        pad_size = items_per_device * total_devices - x.shape[0]

        if pad_size > 0:
            pad_width = [(0, pad_size)] + [(0, 0)] * (x.ndim - 1)
            x_padded = jnp.pad(x, pad_width, mode="constant", constant_values=0)
        else:
            x_padded = x

        local_shard_size = items_per_device * n_local_devices
        xs = jax.device_put(
            jnp.array_split(x_padded[:local_shard_size], n_local_devices),
            mesh.local_devices,
        )

        global_shape = (x_padded.shape[0], *x_padded.shape[1:])
        sharded_array = jax.make_array_from_single_device_arrays(global_shape, sharding, xs)

        return sharded_array

    return shard


class LMEvaluator(lm_eval.api.model.LM):
    _DEFAULT_MAX_LENGTH = 2048
    """Adapting LLM harness for our GPT model."""

    def __init__(self, model, mesh, batch_size=2, max_length=None, truncation=False):
        super().__init__()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        self.truncation = truncation
        self.batch_size = batch_size
        self._max_length = max_length
        self.mesh = mesh
        self.replicated_sharding = NamedSharding(self.mesh, P())
        self.gen_fn = jax.jit(greedy_generate, static_argnums=(2, 3))

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    def loglikelihood(self, requests):
        print("Calling loglikelihood")
        new_reqs = []
        print(f"Processing {len(requests)} requests.")
        for context, continuation in [req.args for req in requests]:
            if context == "":
                raise ValueError("Context cannot be empty.")
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def generate_until(self, requests):
        res = []
        for req in tqdm(requests, desc="generate_until"):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            prompt_ids = self.tok_encode(context)
            prompt = jax.device_put(
                jax.numpy.array(prompt_ids, dtype=jax.numpy.int32),
                self.replicated_sharding,
            )

            # generate step
            full_seq = self.gen_fn(self.model, prompt, max_gen_toks, self.eot_token_id)
            gen_tokens = full_seq[len(prompt_ids):].tolist()

            # trim eot token
            if self.eot_token_id in gen_tokens:
                gen_tokens = gen_tokens[: gen_tokens.index(self.eot_token_id)]

            # trim stop string
            gen_text = self.tokenizer.decode(gen_tokens)
            for stop in until:
                if stop and stop in gen_text:
                    gen_text = gen_text[: gen_text.index(stop)]

            res.append(gen_text)
        return res

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("This function is not implemented yet.")

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        encoding = self.tokenizer.encode(string)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings,
        padding_side="left",
        left_truncate_len=None,
        truncation=False,
    ):
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = False

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)

        n_reordered_requests = len(re_ord)
        batch_size = self.batch_size if self.batch_size != "auto" else override_bs if override_bs is not None else 0
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and n_reordered_requests > 0 and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)))
        for chunk in chunks:
            inps = []
            inplens = []  # length of context + continuation
            max_padded_len = None  # max length of context + continuation
            ctxlens = []
            cntlens = []

            for _, context_enc, continuation_enc in chunk:
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                inp = jnp.array(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :],
                ).astype(jnp.int32)
                (inplen,) = inp.shape

                max_padded_len = max(max_padded_len, inplen) if max_padded_len is not None else inplen

                inps.append(inp)  # [1, inp_length]
                inplens.append(inplen)
                ctxlens.append(len(context_enc))
                cntlens.append(len(continuation_enc))

            call_kwargs = {}
            batched_inps = pad_and_concat(max_padded_len, inps, padding_side="right")  # [batch, padding_len_inp]
            batched_inps = jax.device_put(batched_inps, self.replicated_sharding)
            multi_logits = jax.nn.log_softmax(self._model_call(batched_inps, self.model, **call_kwargs), axis=-1)[
                : len(inps)
            ]

            for inp, inplen, ctxlen, cntlen, logits in zip(inps, inplens, ctxlens, cntlens, multi_logits):
                logprobs = logits[None, :]
                token_logprob = jnp.take_along_axis(
                    logprobs[:, ctxlen - 1 : inplen - 1],
                    inp[:, ctxlen:inplen, None],
                    axis=2,
                )  # [1, seq, 1]
                greedy_tokens = logprobs[:, ctxlen - 1 : inplen - 1].argmax(axis=-1)
                cont_toks = jnp.array(inp[:, ctxlen:inplen], dtype=jnp.int32)[None, :]  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                answer = (float(token_logprob.sum()), bool(max_equal))
                res.append(answer)
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def _select_cont_toks(self, logits, contlen=None, inplen=None):
        assert contlen and inplen, "Must pass input len and cont. len to select scored logits for causal LM"
        logits = logits[inplen - contlen : inplen]

        return logits

    def _encode_pair(self, context: str, continuation: str):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation, add_special_tokens=False)
        context_enc = self.tok_encode(context, add_special_tokens=False)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    @eqx.filter_jit
    def _model_call(self, x_BxT, model, attn_mask=None, labels=None):
        logits_BxTxV = vmap(model)(x_BxT, key=None).astype(jnp.float32)
        return logits_BxTxV

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        stopping_criteria = handle_stop_sequences(stop, self.tokenizer.eos_token)
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

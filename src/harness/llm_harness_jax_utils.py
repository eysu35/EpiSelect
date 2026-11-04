from typing import (
    List,
    Literal,
)

import jax.numpy as jnp


def pad_and_concat(
    max_length: int,
    tensors: List,
    padding_side: Literal["right", "left"] = "right",
):
    assert padding_side == "left" or padding_side == "right", (
        f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'"
    )

    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:
            tensor = tensor.squeeze(0)  # squeeze, in case passed [1, seq] size
        tensor_len = tensor.shape[0]
        if tensor_len < max_length:
            if padding_side == "right":
                tensors[i] = jnp.concatenate(
                    [
                        tensor,  # [seq]
                        jnp.zeros(
                            max_length - tensor_len,
                        ).astype(jnp.int32),  # [padding_length - seq]
                    ],
                    axis=0,
                )[jnp.newaxis, :]
            else:
                # left-pad
                tensors[i] = jnp.concatenate(
                    [
                        jnp.zeros(
                            max_length - tensor_len,
                        ).astype(jnp.int32),  # [padding_length - seq]
                        tensor,  # [seq]
                    ],
                    axis=0,
                )[jnp.newaxis, :]
        else:
            tensors[i] = tensor[jnp.newaxis, :]

    return jnp.concatenate(tensors, axis=0)

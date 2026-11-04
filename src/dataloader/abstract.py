import typing as tp
from abc import ABC, abstractmethod

import jax

Array = jax.Array


class Loader(ABC):
    @abstractmethod
    def __init__(self, split: str, mesh, block_size: int, g_accum_iters: int, **kwargs):
        pass

    @abstractmethod
    def get_batch(self, batch_size: int, itr: int) -> tp.Tuple[Array, Array, tp.Dict]:
        pass

    @abstractmethod
    def update(self, loss_GxB: Array, info: tp.Dict, itr: int) -> tp.Dict:
        pass

    # "host" is per-process (rng, stream position); "global" is identical on every host
    def state_dict(self) -> tp.Dict[str, tp.Dict]:
        return {"host": {}, "global": {}}

    def load_state_dict(self, host: tp.Dict, glob: tp.Dict):
        pass

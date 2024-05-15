import jax
import equinox as eqx
from jaxtyping import Array, Float


class Cell_reaction(eqx.Module):
    layers: list
    CELL_CHANNELS: int
    SIGNAL_CHANNELS: int
    TOTAL_CHANNELS: int

    def __init__(self,CELL_CHANNELS,SIGNAL_CHANNELS,key):
        self.CELL_CHANNELS = CELL_CHANNELS
        self.SIGNAL_CHANNELS = SIGNAL_CHANNELS
        self.TOTAL_CHANNELS = CELL_CHANNELS + SIGNAL_CHANNELS
        key1,key2 = jax.random.split(key,2)
        self.layers = [
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.TOTAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key1
            ),
            jax.nn.relu,
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.CELL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key2
            ),
            jax.nn.tanh
        ]
        where = lambda l:l.weight
        scale = 0.0
        self.layers[-2] = eqx.tree_at(where,
                                      self.layers[-2],
                                      scale*jax.random.normal(key=key2,shape=self.layers[-2].weight.shape))
    def __call__(self,X: Float[Array, "C x y"])->Float[Array,"cell x y"]:
        for L in self.layers:
            X = L(X)
        return X
    def partition(self):
        return eqx.partition(self,eqx.is_array)
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
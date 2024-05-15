import jax
import equinox as eqx
from jaxtyping import Array, Float


class Signal_reaction(eqx.Module):
    production_layers: list
    decay_layers: list
    CELL_CHANNELS: int
    SIGNAL_CHANNELS: int
    TOTAL_CHANNELS: int



    def __init__(self,CELL_CHANNELS,SIGNAL_CHANNELS,key):
        self.CELL_CHANNELS = CELL_CHANNELS
        self.SIGNAL_CHANNELS = SIGNAL_CHANNELS
        self.TOTAL_CHANNELS = CELL_CHANNELS + SIGNAL_CHANNELS
        
        key1,key2,key3,key4 = jax.random.split(key,4)
        self.production_layers = [
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
                out_channels=self.SIGNAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key2
            ),
            jax.nn.relu
        ]
        self.decay_layers = [
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.TOTAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key3
            ),
            jax.nn.relu,
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.SIGNAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key4
            ),
            jax.nn.relu
        ]
        where = lambda l:l.weight
        scale_prod = 0.001
        scale_decay = 0.1
        self.decay_layers[-2] = eqx.tree_at(where,
                                            self.decay_layers[-2],
                                            scale_decay*jax.random.normal(key=key4,shape=self.decay_layers[-2].weight.shape))
        self.production_layers[-2] = eqx.tree_at(where,
                                            self.production_layers[-2],
                                            scale_prod*jax.random.normal(key=key2,shape=self.production_layers[-2].weight.shape))

    def __call__(self,X: Float[Array, "C x y"])->Float[Array,"signal x y"]:
        production = X
        decay = X
        signals=X[self.CELL_CHANNELS:]
        for L in self.production_layers:
            production = L(production)
        
        for L in self.decay_layers:
            decay = L(decay)
        stability_factor = 0.01
        return production - signals*decay - stability_factor*signals
    
    def partition(self):
        return eqx.partition(self,eqx.is_array)
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
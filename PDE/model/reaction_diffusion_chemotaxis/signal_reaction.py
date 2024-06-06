import jax
import equinox as eqx
from jaxtyping import Array, Float


class Signal_reaction(eqx.Module):
    production_layers: list
    decay_layers: list
    CELL_CHANNELS: int
    SIGNAL_CHANNELS: int
    TOTAL_CHANNELS: int
    STABILITY_FACTOR: float



    def __init__(self,CELL_CHANNELS,SIGNAL_CHANNELS,INTERNAL_ACTIVATION,OUTER_ACTIVATION,INIT_SCALE,STABILITY_FACTOR,key):
        self.CELL_CHANNELS = CELL_CHANNELS
        self.SIGNAL_CHANNELS = SIGNAL_CHANNELS
        self.TOTAL_CHANNELS = CELL_CHANNELS + SIGNAL_CHANNELS
        self.STABILITY_FACTOR = STABILITY_FACTOR
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
            INTERNAL_ACTIVATION,
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.SIGNAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key2
            ),
            OUTER_ACTIVATION
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
            INTERNAL_ACTIVATION,
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.SIGNAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key4
            ),
            OUTER_ACTIVATION
        ]
        where = lambda l:l.weight


        self.decay_layers[0] = eqx.tree_at(where,
                                            self.decay_layers[0],
                                            INIT_SCALE*jax.random.normal(key=key1,shape=self.decay_layers[0].weight.shape))
        self.production_layers[0] = eqx.tree_at(where,
                                            self.production_layers[0],
                                            INIT_SCALE*jax.random.normal(key=key2,shape=self.production_layers[0].weight.shape))

        self.decay_layers[-2] = eqx.tree_at(where,
                                            self.decay_layers[-2],
                                            INIT_SCALE*jax.random.normal(key=key3,shape=self.decay_layers[-2].weight.shape))
        self.production_layers[-2] = eqx.tree_at(where,
                                            self.production_layers[-2],
                                            INIT_SCALE*jax.random.normal(key=key4,shape=self.production_layers[-2].weight.shape))

    def __call__(self,X: Float[Array, "C x y"])->Float[Array,"signal x y"]:
        production = X
        decay = X
        signals=X[self.CELL_CHANNELS:]
        for L in self.production_layers:
            production = L(production)
        
        for L in self.decay_layers:
            decay = L(decay)
        
        return production - signals*decay - self.STABILITY_FACTOR*signals
    
    def partition(self):
        return eqx.partition(self,eqx.is_array)
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
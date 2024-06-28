import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class Signal_reaction(eqx.Module):
    production_layers: list
    decay_layers: list
    CELL_CHANNELS: int
    SIGNAL_CHANNELS: int
    TOTAL_CHANNELS: int
    STABILITY_FACTOR: float




    def __init__(self,CELL_CHANNELS,SIGNAL_CHANNELS,INTERNAL_ACTIVATION,OUTER_ACTIVATION,INIT_SCALE,USE_BIAS,STABILITY_FACTOR,key):
        self.CELL_CHANNELS = CELL_CHANNELS
        self.SIGNAL_CHANNELS = SIGNAL_CHANNELS
        self.TOTAL_CHANNELS = CELL_CHANNELS + SIGNAL_CHANNELS
        self.STABILITY_FACTOR = STABILITY_FACTOR
        key1,key2,key3,key4,key5,key6,key7,key8 = jax.random.split(key,8)
        self.production_layers = [
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.TOTAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=USE_BIAS,
                key=key1
            ),
            INTERNAL_ACTIVATION,
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.SIGNAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=USE_BIAS,
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
                use_bias=USE_BIAS,
                key=key3
            ),
            INTERNAL_ACTIVATION,
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.SIGNAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=USE_BIAS,
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
        

        if USE_BIAS:
            where_b = lambda l:l.bias
            self.decay_layers[0] = eqx.tree_at(where_b,
                                                self.decay_layers[0],
                                                INIT_SCALE*jax.random.normal(key=key5,shape=self.decay_layers[0].bias.shape))
            self.production_layers[0] = eqx.tree_at(where_b,
                                                self.production_layers[0],
                                                INIT_SCALE*jax.random.normal(key=key6,shape=self.production_layers[0].bias.shape))

            self.decay_layers[-2] = eqx.tree_at(where_b,
                                                self.decay_layers[-2],
                                                INIT_SCALE*jax.random.normal(key=key7,shape=self.decay_layers[-2].bias.shape))
            self.production_layers[-2] = eqx.tree_at(where_b,
                                                self.production_layers[-2],
                                                INIT_SCALE*jax.random.normal(key=key8,shape=self.production_layers[-2].bias.shape))

    def __call__(self,X: Float[Array, "C x y"])->Float[Array,"signal x y"]:
        production = X
        decay = X
        signals=X[self.CELL_CHANNELS:]
        for L in self.production_layers:
            production = L(production)
        
        for L in self.decay_layers:
            decay = L(decay)
        
        return production - signals*decay - self.STABILITY_FACTOR*signals**3#signals*jnp.max(jnp.abs(signals))*self.STABILITY_FACTOR#self.STABILITY_FACTOR*signals
    
    def partition(self):
        return eqx.partition(self,eqx.is_array)
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
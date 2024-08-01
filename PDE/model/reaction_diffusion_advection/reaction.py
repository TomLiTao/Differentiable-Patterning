import jax
import equinox as eqx
import jax.numpy as jnp
import time
from jaxtyping import Array, Float
from Common.model.custom_functions import construct_polynomials

class R(eqx.Module):
    production_layers: list
    decay_layers: list
    N_CHANNELS: int
    STABILITY_FACTOR: float
    ORDER: int
    polynomial_preprocess: callable
    def __init__(self,
                 N_CHANNELS,
                 INTERNAL_ACTIVATION,
                 OUTER_ACTIVATION,
                 INIT_SCALE,
                 USE_BIAS,
                 STABILITY_FACTOR,
                 ORDER,
                 ZERO_INIT=True,
                 key=jax.random.PRNGKey(int(time.time()))):
        self.STABILITY_FACTOR = STABILITY_FACTOR
        keys = jax.random.split(key,8)
        self.N_CHANNELS = N_CHANNELS
        self.ORDER = ORDER
        N_FEATURES = len(construct_polynomials(jnp.zeros((N_CHANNELS,)),self.ORDER))
        _v_poly = jax.vmap(lambda x: construct_polynomials(x,self.ORDER),in_axes=1,out_axes=1)
        self.polynomial_preprocess = jax.vmap(_v_poly,in_axes=1,out_axes=1)


        self.production_layers = [
            eqx.nn.Conv2d(
                in_channels=N_FEATURES,
                out_channels=N_FEATURES,
                kernel_size=1,
                padding=0,
                use_bias=USE_BIAS,
                key=keys[0]
            ),
            INTERNAL_ACTIVATION,
            eqx.nn.Conv2d(
                in_channels=N_FEATURES,
                out_channels=self.N_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=USE_BIAS,
                key=keys[1]
            ),
            OUTER_ACTIVATION
        ]
        self.decay_layers = [
            eqx.nn.Conv2d(
                in_channels=N_FEATURES,
                out_channels=N_FEATURES,
                kernel_size=1,
                padding=0,
                use_bias=USE_BIAS,
                key=keys[2]
            ),
            INTERNAL_ACTIVATION,
            eqx.nn.Conv2d(
                in_channels=N_FEATURES,
                out_channels=self.N_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=USE_BIAS,
                key=keys[3]
            ),
            OUTER_ACTIVATION
        ]
        where = lambda l:l.weight
        self.decay_layers[0] = eqx.tree_at(where,
                                            self.decay_layers[0],
                                            INIT_SCALE*jax.random.normal(key=keys[0],shape=self.decay_layers[0].weight.shape))
        self.production_layers[0] = eqx.tree_at(where,
                                            self.production_layers[0],
                                            INIT_SCALE*jax.random.normal(key=keys[1],shape=self.production_layers[0].weight.shape))

        self.decay_layers[-2] = eqx.tree_at(where,
                                            self.decay_layers[-2],
                                            INIT_SCALE*jax.random.normal(key=keys[2],shape=self.decay_layers[-2].weight.shape))
        self.production_layers[-2] = eqx.tree_at(where,
                                            self.production_layers[-2],
                                            INIT_SCALE*jax.random.normal(key=keys[3],shape=self.production_layers[-2].weight.shape))
        

        if USE_BIAS:
            where_b = lambda l:l.bias
            self.decay_layers[0] = eqx.tree_at(where_b,
                                                self.decay_layers[0],
                                                INIT_SCALE*jax.random.normal(key=keys[4],shape=self.decay_layers[0].bias.shape))
            self.production_layers[0] = eqx.tree_at(where_b,
                                                self.production_layers[0],
                                                INIT_SCALE*jax.random.normal(key=keys[5],shape=self.production_layers[0].bias.shape))

            self.decay_layers[-2] = eqx.tree_at(where_b,
                                                self.decay_layers[-2],
                                                INIT_SCALE*jax.random.normal(key=keys[6],shape=self.decay_layers[-2].bias.shape))
            self.production_layers[-2] = eqx.tree_at(where_b,
                                                self.production_layers[-2],
                                                INIT_SCALE*jax.random.normal(key=keys[7],shape=self.production_layers[-2].bias.shape))


        if ZERO_INIT:
            w_where = lambda l: l.weight
            self.production_layers[-2] = eqx.tree_at(w_where,
                                                     self.production_layers[-2],
                                                     jnp.zeros(self.production_layers[-2].weight.shape))
            self.decay_layers[-2] = eqx.tree_at(w_where,
                                                self.decay_layers[-2],
                                                jnp.zeros(self.decay_layers[-2].weight.shape))
            if USE_BIAS:
                b_where = lambda l: l.bias
                self.production_layers[-2] = eqx.tree_at(b_where,
                                                         self.production_layers[-2],
                                                         jnp.zeros(self.production_layers[-2].bias.shape))
                self.decay_layers[-2] = eqx.tree_at(b_where,
                                                    self.decay_layers[-2],
                                                    jnp.zeros(self.decay_layers[-2].bias.shape))



    def __call__(self,X: Float[Array, "{self.N_CHANNELS} x y"])->Float[Array,"{self.N_CHANNELS} x y"]:
        X_poly = self.polynomial_preprocess(X)
        print(f"Reaction shape: {X_poly.shape}")
        production = X_poly
        decay = X_poly
        for L in self.production_layers:
            production = L(production)
        
        for L in self.decay_layers:
            decay = L(decay)
        
        return production - X*decay - self.STABILITY_FACTOR*X**3#signals*jnp.max(jnp.abs(signals))*self.STABILITY_FACTOR#self.STABILITY_FACTOR*signals
    
    def partition(self):
        return eqx.partition(self,eqx.is_array)
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
        
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import time
from jaxtyping import Array, Float
from Common.model.custom_functions import construct_polynomials,set_layer_weights
from einops import rearrange,repeat

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
                 INIT_TYPE,
                 USE_BIAS,
                 STABILITY_FACTOR,
                 ORDER,
                 N_LAYERS,
                 ZERO_INIT=True,
                 key=jax.random.PRNGKey(int(time.time()))):
        self.STABILITY_FACTOR = STABILITY_FACTOR
        keys = jax.random.split(key,4*(N_LAYERS+1))
        self.N_CHANNELS = N_CHANNELS
        self.ORDER = ORDER
        N_FEATURES = len(construct_polynomials(jnp.zeros((N_CHANNELS,)),self.ORDER))
        _v_poly = jax.vmap(lambda x: construct_polynomials(x,self.ORDER),in_axes=1,out_axes=1)
        self.polynomial_preprocess = jax.vmap(_v_poly,in_axes=1,out_axes=1)

        _inner_layers_p = [eqx.nn.Conv2d(in_channels=N_FEATURES,out_channels=self.N_CHANNELS,kernel_size=1,padding=0,use_bias=USE_BIAS,key=key) for key in keys[:N_LAYERS]]
        _inner_activations = [lambda x:INTERNAL_ACTIVATION(self.polynomial_preprocess(x)) for _ in range(N_LAYERS)]
        self.production_layers = _inner_layers_p+_inner_activations
        self.production_layers[::2] = _inner_layers_p
        self.production_layers[1::2] = _inner_activations
        self.production_layers.append(eqx.nn.Conv2d(in_channels=N_FEATURES,out_channels=self.N_CHANNELS,kernel_size=1,padding=0,use_bias=USE_BIAS,key=keys[2*N_LAYERS]))
        self.production_layers.append(OUTER_ACTIVATION)

        _inner_layers_d = [eqx.nn.Conv2d(in_channels=N_FEATURES,out_channels=self.N_CHANNELS,kernel_size=1,padding=0,use_bias=USE_BIAS,key=key) for key in keys[N_LAYERS:2*N_LAYERS]]
        self.decay_layers = _inner_layers_d+_inner_activations
        self.decay_layers[::2] = _inner_layers_d
        self.decay_layers[1::2] = _inner_activations
        self.decay_layers.append(eqx.nn.Conv2d(in_channels=N_FEATURES,out_channels=self.N_CHANNELS,kernel_size=1,padding=0,use_bias=USE_BIAS,key=keys[2*N_LAYERS+1]))
        self.decay_layers.append(OUTER_ACTIVATION)



        where = lambda l:l.weight
        where_b = lambda l:l.bias


        for i in range(0,len(self.production_layers)//2):
            self.production_layers[2*i] = eqx.tree_at(where,
                                                    self.production_layers[2*i],
                                                    set_layer_weights(self.production_layers[2*i].weight.shape,keys[i],INIT_TYPE,INIT_SCALE))
            self.decay_layers[2*i] = eqx.tree_at(where,
                                               self.decay_layers[2*i],
                                               set_layer_weights(self.decay_layers[2*i].weight.shape,keys[i+len(self.production_layers)//2],INIT_TYPE,INIT_SCALE))
            if USE_BIAS:
                self.production_layers[2*i] = eqx.tree_at(where_b,
                                                        self.production_layers[2*i],
                                                        INIT_SCALE*jax.random.normal(key=keys[i+len(self.production_layers)],shape=self.production_layers[2*i].bias.shape))
                self.decay_layers[2*i] = eqx.tree_at(where_b,
                                                    self.decay_layers[2*i],
                                                    INIT_SCALE*jax.random.normal(key=keys[i+3*len(self.production_layers)//2],shape=self.decay_layers[2*i].bias.shape))
        

        if ZERO_INIT:
            self.production_layers[-2] = eqx.tree_at(where,
                                                     self.production_layers[-2],
                                                     jnp.zeros(self.production_layers[-2].weight.shape))
            self.decay_layers[-2] = eqx.tree_at(where,
                                                self.decay_layers[-2],
                                                jnp.zeros(self.decay_layers[-2].weight.shape))
            if USE_BIAS:
                self.production_layers[-2] = eqx.tree_at(where_b,
                                                         self.production_layers[-2],
                                                         jnp.zeros(self.production_layers[-2].bias.shape))
                self.decay_layers[-2] = eqx.tree_at(where_b,
                                                    self.decay_layers[-2],
                                                    jnp.zeros(self.decay_layers[-2].bias.shape))



    def __call__(self,X: Float[Array, "{self.N_CHANNELS} x y"])->Float[Array,"{self.N_CHANNELS} x y"]:
        X_poly = self.polynomial_preprocess(X)
        #print(f"Reaction shape: {X_poly.shape}")
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
        
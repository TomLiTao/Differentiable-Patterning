import jax
import equinox as eqx
import jax.numpy as jnp
import time
from Common.model.spatial_operators import Ops
from jaxtyping import Array, Float
from Common.model.custom_functions import construct_polynomials
import jax.random as jr

class D(eqx.Module):
    layers: list
    ops: eqx.Module
    N_CHANNELS: int
    PADDING: str
    ORDER: int
    polynomial_preprocess: callable
    def __init__(self,
                 N_CHANNELS,
                 PADDING,
                 dx,
                 INTERNAL_ACTIVATION,
                 OUTER_ACTIVATION,
                 INIT_SCALE,
                 USE_BIAS,
                 ORDER,
                 ZERO_INIT,
                 key):
        self.N_CHANNELS = N_CHANNELS
        self.ORDER = ORDER
        N_FEATURES = len(construct_polynomials(jnp.zeros((N_CHANNELS,)),self.ORDER))
        _v_poly = jax.vmap(lambda x: construct_polynomials(x,self.ORDER),in_axes=1,out_axes=1)
        self.polynomial_preprocess = jax.vmap(_v_poly,in_axes=1,out_axes=1)
        self.PADDING = PADDING
        keys = jr.split(key,2)
        self.layers = [eqx.nn.Conv2d(in_channels=N_FEATURES,
                                     out_channels=N_FEATURES,
                                     kernel_size=1,
                                     padding=0,
                                     use_bias=USE_BIAS,
                                     key=keys[0]),
                        INTERNAL_ACTIVATION,
                        eqx.nn.Conv2d(in_channels=N_FEATURES,
                                     out_channels=self.N_CHANNELS,
                                     kernel_size=1,
                                     padding=0,
                                     use_bias=USE_BIAS,
                                     key=keys[1]),
                        OUTER_ACTIVATION ]
        

        self.ops = Ops(PADDING=PADDING,dx=dx)
        w_where = lambda l: l.weight
        self.layers[0] = eqx.tree_at(w_where,
                                     self.layers[0],
                                     INIT_SCALE*jr.normal(keys[0],self.layers[0].weight.shape))
        self.layers[2] = eqx.tree_at(w_where,
                                     self.layers[2],
                                     INIT_SCALE*jr.normal(keys[1],self.layers[2].weight.shape))
        
        if USE_BIAS:
            b_where = lambda l: l.bias
            self.layers[0] = eqx.tree_at(b_where,
                                         self.layers[0],
                                         INIT_SCALE*jr.normal(keys[2],self.layers[0].bias.shape))
            self.layers[2] = eqx.tree_at(b_where,
                                         self.layers[2],
                                         INIT_SCALE*jr.normal(keys[3],self.layers[2].bias.shape))
            
        if ZERO_INIT:
            self.layers[0] = eqx.tree_at(w_where,
                                         self.layers[0],
                                         jnp.zeros(self.layers[0].weight.shape))
            self.layers[2] = eqx.tree_at(w_where,
                                         self.layers[2],
                                         jnp.zeros(self.layers[2].weight.shape))
            if USE_BIAS:
                self.layers[0] = eqx.tree_at(b_where,
                                             self.layers[0],
                                             jnp.zeros(self.layers[0].bias.shape))
                self.layers[2] = eqx.tree_at(b_where,
                                             self.layers[2],
                                             jnp.zeros(self.layers[2].bias.shape))
    @eqx.filter_jit
    def __call__(self,X: Float[Array, "{self.N_CHANNELS} x y"])->Float[Array, "{self.N_CHANNELS} x y"]:
        Dx = self.polynomial_preprocess(X)
        print(f"Diffusion shape: {Dx.shape}")
        for L in self.layers:
            Dx = L(Dx)
        return self.ops.NonlinearDiffusion(Dx,X)
    def partition(self):
        total_diff,total_static = eqx.partition(self,eqx.is_array)
        op_diff,op_static = self.ops.partition()
        where_ops = lambda m: m.ops
        total_diff = eqx.tree_at(where_ops,total_diff,op_diff)
        total_static = eqx.tree_at(where_ops,total_static,op_static)
        return total_diff,total_static
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)

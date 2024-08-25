"""
Function that handles the advection bit. Function call does div(V(X)*X), trainable parameters are attached to V only.
V: Re^c->Re^(2c)
"""
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import time
from Common.model.spatial_operators import Ops
from einops import rearrange,repeat
from jaxtyping import Array, Float
from Common.model.custom_functions import construct_polynomials
from jaxtyping import Array, Float
class V(eqx.Module):
    layers: list
    N_CHANNELS: int
    DIM: int
    ops: eqx.Module
    ORDER: int
    polynomial_preprocess: callable
    def __init__(self,
                 N_CHANNELS,
                 PADDING,
                 dx,
                 INTERNAL_ACTIVATION,
                 OUTER_ACTIVATION,
                 INIT_SCALE,
                 INIT_TYPE,
                 USE_BIAS,
                 ORDER,
                 N_LAYERS,
                 ZERO_INIT=True,
                 DIM=2,
                 key=jax.random.PRNGKey(int(time.time()))):
        #keys = jr.split(key,4)
        self.N_CHANNELS = N_CHANNELS
        self.ORDER = ORDER
        N_FEATURES = len(construct_polynomials(jnp.zeros((N_CHANNELS,)),self.ORDER))
        _v_poly = jax.vmap(lambda x: construct_polynomials(x,self.ORDER),in_axes=1,out_axes=1)
        self.polynomial_preprocess = jax.vmap(_v_poly,in_axes=1,out_axes=1)
        

        self.DIM = DIM
        self.ops = Ops(PADDING=PADDING,dx=dx)

        keys = jr.split(key,2*(N_LAYERS+1))
        _inner_layers = [eqx.nn.Conv2d(in_channels=N_FEATURES,out_channels=self.N_CHANNELS,kernel_size=1,padding=0,use_bias=USE_BIAS,key=key) for key in keys[:N_LAYERS]]
        _inner_activations = [lambda x:INTERNAL_ACTIVATION(self.polynomial_preprocess(x)) for _ in range(N_LAYERS)]
        self.layers = _inner_layers + _inner_activations
        self.layers[::2] = _inner_layers
        self.layers[1::2] = _inner_activations
        self.layers.append(eqx.nn.Conv2d(in_channels=N_FEATURES,out_channels=self.DIM*self.N_CHANNELS,kernel_size=1,padding=0,use_bias=USE_BIAS,key=keys[N_LAYERS]))
        self.layers.append(OUTER_ACTIVATION)
        
        
        
        
        
        
        # self.layers = [eqx.nn.Conv2d(in_channels=N_FEATURES,
        #                              out_channels=N_FEATURES,
        #                              kernel_size=1,
        #                              padding=0,
        #                              use_bias=USE_BIAS,
        #                              key=keys[0]),
        #                INTERNAL_ACTIVATION,
        #                eqx.nn.Conv2d(in_channels=N_FEATURES,
        #                              out_channels=self.DIM*self.N_CHANNELS,
        #                              kernel_size=1,
        #                              padding=0,
        #                              use_bias=USE_BIAS,
        #                              key=keys[1]),
        #                 OUTER_ACTIVATION]
        
        w_where = lambda l: l.weight
        b_where = lambda l: l.bias

        def set_layer_weights(shape,key):
            if INIT_TYPE=="orthogonal":
                r = jr.orthogonal(key,n=max(shape[0],shape[1]))
                r = r[:shape[0],:shape[1]]
                r = repeat(r,"i j -> i j () ()")
                return INIT_SCALE*r
            if INIT_TYPE=="normal":
                return INIT_SCALE*jr.normal(key,shape)
            if INIT_TYPE=="diagonal":
                a = 0.9
                i = jnp.eye(shape[0],shape[1])
                i = repeat(i,"i j -> i j () ()")
                r = jr.normal(key,shape)
                return INIT_SCALE*(a*i+(1-a)*r)
            if INIT_TYPE=="permuted":
                a = 0.9
                i = jr.permutation(key,jnp.eye(shape[0],shape[1]))
                i = repeat(i,"i j -> i j () ()")
                r = jr.normal(key,shape)
                return INIT_SCALE*(a*i+(1-a)*r)
        
        


        for i in range(0,len(self.layers)//2):
            self.layers[2*i] = eqx.tree_at(w_where,
                                           self.layers[2*i],
                                           set_layer_weights(self.layers[2*i].weight.shape,keys[i]))
            if USE_BIAS:
                self.layers[2*i] = eqx.tree_at(b_where,
                                               self.layers[2*i],
                                               INIT_SCALE*jr.normal(keys[i+len(self.layers)],self.layers[2*i].bias.shape))
        self.layers[-2] = eqx.tree_at(w_where,
                                      self.layers[-2],
                                      jnp.concatenate((self.layers[-2].weight[:self.N_CHANNELS],
                                                 self.layers[-2].weight[:self.N_CHANNELS]),axis=0))
        # self.layers[0] = eqx.tree_at(w_where,
        #                              self.layers[0],
        #                              #INIT_SCALE*jr.normal(keys[0],self.layers[0].weight.shape))
        #                              set_layer_weights(self.layers[0].weight.shape,keys[0]))
        # self.layers[2] = eqx.tree_at(w_where,
        #                              self.layers[2],
        #                              #INIT_SCALE*jr.normal(keys[1],self.layers[2].weight.shape))
        #                              set_layer_weights(self.layers[2].weight.shape,keys[1]))
        
        # if USE_BIAS:
        #     b_where = lambda l: l.bias
        #     self.layers[0] = eqx.tree_at(b_where,
        #                                  self.layers[0],
        #                                  INIT_SCALE*jr.normal(keys[2],self.layers[0].bias.shape))
        #     self.layers[2] = eqx.tree_at(b_where,
        #                                  self.layers[2],
        #                                  INIT_SCALE*jr.normal(keys[3],self.layers[2].bias.shape))
        
        if ZERO_INIT:               
            self.layers[-2] = eqx.tree_at(w_where,self.layers[-2],jnp.zeros(self.layers[-2].weight.shape))
            if USE_BIAS:
                self.layers[-2] = eqx.tree_at(b_where,self.layers[-2],jnp.zeros(self.layers[-2].bias.shape))

        
    @eqx.filter_jit
    def f(self,X: Float[Array, "{self.N_CHANNELS} x y"])->Float[Array,"{self.N_CHANNELS} x y"]:
        X = self.polynomial_preprocess(X)
        #print(f"Advection shape: {X.shape}")
        for L in self.layers:
            X = L(X)
        return X
    @eqx.filter_jit
    def __call__(self,X: Float[Array, "{self.N_CHANNELS} x y"])->Float[Array,"{self.N_CHANNELS} x y"]:
        vx = self.f(X)
        X = repeat(X,"c x y -> dim c x y",dim=self.DIM)
        vx = rearrange(vx,"(dim c) x y -> dim c x y",dim=self.DIM)
        return self.ops.Div(vx*X)
        
    
    def partition(self):
        total_diff,total_static = eqx.partition(self,eqx.is_array)
        op_diff,op_static = self.ops.partition()
        where_ops = lambda m: m.ops
        total_diff = eqx.tree_at(where_ops,total_diff,op_diff)
        total_static = eqx.tree_at(where_ops,total_static,op_static)
        return total_diff,total_static
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
        
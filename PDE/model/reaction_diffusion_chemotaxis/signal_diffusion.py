import jax
import equinox as eqx
import einops
from jaxtyping import Array, Float
#from Common.model.laplacian import Laplacian
from Common.model.spatial_operators import Ops

class Signal_diffusion(eqx.Module):
    diffusion_constants: list
    ops: eqx.Module
    SIGNAL_CHANNELS: int
    CELL_CHANNELS: int

    def __init__(self,CELL_CHANNELS,SIGNAL_CHANNELS,PADDING,dx,key):
        self.SIGNAL_CHANNELS=SIGNAL_CHANNELS
        self.CELL_CHANNELS=CELL_CHANNELS
        self.diffusion_constants = eqx.nn.Conv2d(in_channels=self.SIGNAL_CHANNELS,
                                                 out_channels=self.SIGNAL_CHANNELS,
                                                 kernel_size=1,
                                                 use_bias=False,
                                                 key=key,
                                                 groups=self.SIGNAL_CHANNELS)
        
        
        
        self.ops = Ops(PADDING=PADDING,dx=dx)
        where = lambda l: l.weight
        self.diffusion_constants= eqx.tree_at(where,self.diffusion_constants,jax.numpy.abs(self.diffusion_constants.weight))

    def __call__(self,X: Float[Array, "C x y"])->Float[Array, "signal x y"]:
        signals=X[self.CELL_CHANNELS:]
        return self.diffusion_constants(self.ops.Lap(signals))
    
    def partition(self):
        total_diff,total_static = eqx.partition(self,eqx.is_array)
        op_diff,op_static = self.ops.partition()
        where_ops = lambda m: m.ops
        total_diff = eqx.tree_at(where_ops,total_diff,op_diff)
        total_static = eqx.tree_at(where_ops,total_static,op_static)
        return total_diff,total_static
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
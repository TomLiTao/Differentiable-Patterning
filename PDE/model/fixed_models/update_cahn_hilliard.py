import jax
import equinox as eqx
import jax.numpy as jnp
import time
from jaxtyping import Array, Float, PyTree, Scalar
from einops import rearrange

from Common.model.spatial_operators import Ops
class F(eqx.Module):
    ops: Ops
    gamma: float
    D: float
    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE,
                 gamma = 1.0,
                 D = 0.1
                 ):
        """Implementation of basic pattern formating cahn-hilliard model

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
            dx (float): length scale for operators
            gamma (float, optional): structure lengthscale. Defaults to 1.0.
            D (float, optional): Diffusion strength. Defaults to 0.1.
        """
        
        self.gamma = gamma
        self.D = D
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"1 x y"],
                 args)->Float[Scalar, "1 x y"]:
        
        mu = X**3 - X - self.gamma*self.ops.Lap(X)
        return self.D*self.ops.Lap(mu)
    

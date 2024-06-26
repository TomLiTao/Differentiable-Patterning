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
    alpha: float
    DA: float
    DB: float
    epsilon: float
    def __init__(self,
                 PADDING,
                 dx,
                 KERNEL_SCALE,
                 DA=0.1,
                 DB=0.05,
                 alpha=0.06230,
                 gamma=0.06268):
        """Implementation of basic pattern formation model from figure 4 in Hillen & Painter "A users guide to PDE models for chemotaxis"

        Args:
            PADDING (str): Boundary type: 'ZEROS', 'REFLECT', 'REPLICATE' or 'CIRCULAR'
            dx (float): _description_
            logistic_growth_rate (float, optional): _description_. Defaults to 0.1.
            gamma (float, optional): _description_. Defaults to 10.0.
            alpha (float, optional): _description_. Defaults to 0.5.
            chi (float, optional): _description_. Defaults to 5.0.
            D (float, optional): _description_. Defaults to 0.1.
        """
        
        self.gamma=gamma
        self.alpha=alpha
        self.DA = DA
        self.DB = DB        
        self.ops = Ops(PADDING,dx,KERNEL_SCALE)
        self.epsilon = 1e-4

    def __call__(self,
                 t: Float[Scalar, ""],
                 X: Float[Scalar,"2 x y"],
                 args)->Float[Scalar, "2 x y"]:
        
        A = X[0:1]
        B = X[1:2]
        
        dA = self.DA*self.ops.Lap(A) - A*B*B + self.alpha*(1-A)
        dB = self.DB*self.ops.Lap(B) + A*B*B - (self.gamma + self.alpha)*B
        
        return jnp.concatenate((dA,dB),axis=0)
    

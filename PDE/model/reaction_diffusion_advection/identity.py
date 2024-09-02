import jax
import equinox as eqx

from jaxtyping import Array, Float

class I(eqx.Module):
    TYPE: str
    def __init__(self,TYPE):
        self.TYPE = TYPE

        



    def __call__(self,X):
        
        return 0
    
    def partition(self):
        return eqx.partition(self,eqx.is_array)
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
        
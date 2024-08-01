import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange
from jaxtyping import Array, Float, Int, Scalar
from itertools import combinations_with_replacement

"""
A collection of useful functions that are reused throughout PDE and NCA models
"""

@eqx.filter_jit
def construct_polynomials(X:Float[Array, "C"],max_power: Int[Scalar, ""])->Float[Array, "K"]:
    """
    Takes the vector X, and returns a longer vector including all polynomials of componentes of X, up to power
    """
    if max_power == 1:
        return X
    else:
        n = X.shape[0]
        terms = []

        for power in range(1, max_power + 1):
            for combo in combinations_with_replacement(range(n), power):
                indices = jnp.array(combo)
                term = jnp.prod(X[indices])
                terms.append(term)
        
        return jnp.array(terms)

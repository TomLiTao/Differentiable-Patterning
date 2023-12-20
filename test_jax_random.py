import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
a = jnp.zeros(2).dtype=="float64"
print(a)
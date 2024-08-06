import jax
import jax.numpy as jnp
import equinox as eqx



def check_training_diverged(mean_loss,x,step,loss_thresh=1e16):
    error = 0
    if jnp.isnan(mean_loss):
        error = 1
        print("|-|-|-|-|-|-  Loss reached NaN at step "+str(step)+" -|-|-|-|-|-|")
    elif any(list(map(lambda x: jnp.any(jnp.isnan(x)), x))):
        error = 2
        print("|-|-|-|-|-|-  X reached NaN at step "+str(step)+" -|-|-|-|-|-|")
    elif mean_loss > loss_thresh:
        error = 3
        print( "|-|-|-|-|-|-  Loss exceded "+str(loss_thresh)+" at step "+str(step)+", optimisation probably diverging  -|-|-|-|-|-|")
    return error

				
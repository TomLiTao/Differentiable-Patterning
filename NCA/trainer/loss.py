import jax.numpy as jnp
import jax
#from eqxvision.models import alexnet
#from eqxvision.utils import CLASSIFICATION_URLS
import equinox as eqx
from lpips_j.lpips import LPIPS
#import eqxvision as eqv
"""
	General format of loss functions here:

	Parameters
	----------
	x : float32 [...,CHANNELS,WIDTH,HEIGHT]
		predictions
	y : float32 [...,CHANNELS,WIDTH,HEIGHT]
		true data

	Returns
	-------
	loss : float32 array [...]
		loss reduced over channel and spatial axes

"""
#loaded_alexnet = alexnet(torch_weights=CLASSIFICATION_URLS['alexnet'])
#loaded_vgg11 = eqv.models.vgg11(torch_weights=CLASSIFICATION_URLS["vgg11"])
lpips = LPIPS()

@jax.jit
def l2(x,y):

	return jnp.sum(((x-y)**2),axis=[-1,-2,-3])
@jax.jit
def euclidean(x,y):
	return jnp.sqrt(jnp.mean(((x-y)**2),axis=[-1,-2,-3]))

#@jax.jit
def random_sampled_euclidean(x,y,key,SAMPLES=64):
	x_r = jnp.einsum("ncxy->cxyn",x)
	y_r = jnp.einsum("ncxy->cxyn",y)
	x_sub = jax.random.choice(key,x_r.reshape((-1,x_r.shape[-1])),(SAMPLES,),False)
	y_sub = jax.random.choice(key,y_r.reshape((-1,y_r.shape[-1])),(SAMPLES,),False)
	return jnp.sqrt(jnp.mean((x_sub-y_sub)**2,axis=0))


# @eqx.filter_jit
# def alexnet(x,y, key):
# 	"""
# 	NOTE THAT CHANNELS IS TRUNCATED TO 3

# 	Parameters
# 	----------
# 	x : float32 [N,CHANNELS,WIDTH,HEIGHT]
# 		predictions
# 	y : float32 [N,CHANNELS,WIDTH,HEIGHT]
# 		true data
# 	key : jax.random.PRNGKey
# 		Jax random number key. 

# 	Returns
# 	-------
# 	loss : float32 [N]

# 	"""
# 	x = jnp.einsum("ncxy->nxyc",x)[...,:3]
# 	y = jnp.einsum("ncxy->nxyc",y)[...,:3]
# 	
# 	lpips = LPIPS(net="alex")
# 	
# 	params = lpips.init(key, x, y)
# 	loss = lpips.apply(params, x, y)
# 	return loss
	
        
@eqx.filter_jit
def vgg(x,y, key):
	"""
	NOTE THAT CHANNELS IS TRUNCATED TO 3

	Parameters
	----------
	x : float32 [N,CHANNELS,WIDTH,HEIGHT]
		predictions
	y : float32 [N,CHANNELS,WIDTH,HEIGHT]
		true data
	key : jax.random.PRNGKey
		Jax random number key. 

	Returns
	-------
	loss : float32 [N]

	"""
	x = jnp.einsum("ncxy->nxyc",x)[...,:3]
	y = jnp.einsum("ncxy->nxyc",y)[...,:3]
	
	
	
	params = lpips.init(key, x, y)
	loss = lpips.apply(params, x, y)
	return loss
	

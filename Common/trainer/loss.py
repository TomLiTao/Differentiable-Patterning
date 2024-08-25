import jax.numpy as jnp
import jax
#from ott.geometry import pointcloud
#from ott.tools import sinkhorn_divergence
#from ott.problems.linear import linear_problem
#from ott.solvers.linear import sinkhorn
#from eqxvision.models import alexnet
#from eqxvision.utils import CLASSIFICATION_URLS
import equinox as eqx
from lpips_j.lpips import LPIPS
#import eqxvision as eqv

#loaded_alexnet = alexnet(torch_weights=CLASSIFICATION_URLS['alexnet'])
#loaded_vgg11 = eqv.models.vgg11(torch_weights=CLASSIFICATION_URLS["vgg11"])
lpips = LPIPS()

@jax.jit
def l2(x,y,key=None):
	"""
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
	return jnp.mean((x-y)**2,axis=[-1,-2,-3])
@jax.jit
def l1(x,y,key=None):
	"""
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
	return jnp.mean(jnp.abs(x-y),axis=[-1,-2,-3])
@jax.jit
def euclidean(x,y,key=None):
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
	return jnp.sqrt(jnp.mean(((x-y)**2),axis=[-1,-2,-3]))

# @jax.jit
# def sinkhorn_divergence_loss(x,y):
# 	"""
# 		Sinkhorn loss - OT distance between 2 point clouds in 2D space

# 		Parameters
# 		----------
# 		x : float32 [N_x,2]
# 			predictions
# 		y : float32 [N_y,2]
# 			true data

# 		Returns
# 		-------
# 		loss : float32 
# 			loss 

# 	"""


# 	geom = pointcloud.PointCloud(x,y)
# 	ot = sinkhorn_divergence.sinkhorn_divergence(
# 		geom,
# 		x=geom.x,
# 		y=geom.y,
# 		static_b=True,
# 	)
# 	return ot.divergence
# 	# ot = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))
# 	# return ot.reg_ot_cost
	
	


@jax.jit
def random_sampled_euclidean(x,y,key,SAMPLES=16):
	x_r = jnp.einsum("ncxy->cxyn",x)
	y_r = jnp.einsum("ncxy->cxyn",y)
	x_sub = jax.random.choice(key,x_r.reshape((-1,x_r.shape[-1])),(SAMPLES,),False)
	y_sub = jax.random.choice(key,y_r.reshape((-1,y_r.shape[-1])),(SAMPLES,),False)
	return jnp.sqrt(jnp.mean((x_sub-y_sub)**2,axis=0))


@jax.jit
def spectral(x,y,key=None):
	""" 
		l2 norm in fourier space (discarding phase information)

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
	fx = jnp.fft.rfft2(x)
	fy = jnp.fft.rfft2(y)
	fx = jnp.abs(fx)
	fy = jnp.abs(fy)
	return l2(fx,fy,key)
        
@jax.jit
def spectral_weighted(x,y,key=None):
	""" 
		l2 norm in fourier space, keeping phase information.
		Weighted to emphasise importance of certain frequencies

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
	fx = jnp.fft.rfft2(x)
	fy = jnp.fft.rfft2(y)
	return jnp.abs(l2(fx,fy,key))
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
	

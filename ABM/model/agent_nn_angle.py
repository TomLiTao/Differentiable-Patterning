import jax
import equinox as eqx
import jax.numpy as jnp
import time

class agent_nn(eqx.Module):
	layers: list
	layers_pheremone: list
	layers_velocity_mag: list
	layers_d_angle: list
	N_CHANNELS: int
	"""
	Individual agent neural network. Reads pheremone conentrations of each channel, convolved with sensor kernels
	i.e. reads value, anterior-posterior and left-right gradients. Also reads orientation of agent in space

	Outputs pheremone increment, velocity magnitude and velocity direction
	"""
	
	def __init__(self,N_CHANNELS,key=jax.random.PRNGKey(int(time.time()))):
		key1,key2,key3,key4,key5,key6,key7 = jax.random.split(key,7)
		self.N_CHANNELS=N_CHANNELS
		def flat_func(x):
			#a = jnp.array(x).flatten()
			x,_ = jax.tree_util.tree_flatten(x)
			x = jnp.concatenate(x)
			#print(x)

			return x
		self.layers = [flat_func,
				       eqx.nn.Linear(in_features=3*self.N_CHANNELS + 2,	# Reads local pheremone concentrations and combines them together
							         out_features=self.N_CHANNELS,
									 use_bias=True,
									 key=key1),
				       jax.nn.relu
# 					   eqx.nn.Linear(in_features=self.N_CHANNELS,	# Reads local pheremone concentrations and combines them together
# 							         out_features=self.N_CHANNELS,
# 									 use_bias=True,
# 									 key=key2),
# 					   jax.nn.relu
					   ]
		self.layers_pheremone = [ 
					   eqx.nn.Linear(in_features=self.N_CHANNELS, 	# Updates the pheremone concentrations at agent position
							         out_features=self.N_CHANNELS,
									 use_bias=True,
									 key=key2),
					   jax.nn.relu,
					    eqx.nn.Linear(in_features=self.N_CHANNELS, 	# Updates the pheremone concentrations at agent position
							         out_features=self.N_CHANNELS,
									 use_bias=True,
									 key=key3) 									# Bounds updates of pheremone between 1 and -1
					   ]	
		self.layers_velocity_mag = [									
					   eqx.nn.Linear(in_features=self.N_CHANNELS, 	# Updates agent velocity 
							         out_features=self.N_CHANNELS,
									 use_bias=False,
									 key=key4),
						jax.nn.relu,
						eqx.nn.Linear(in_features=self.N_CHANNELS, 	# Updates agent velocity 
							         out_features=1,
									 use_bias=False,
									 key=key5),
					   jax.nn.sigmoid 	# Keeps agent speed non negative
					   ]
		self.layers_d_angle = [									
					   eqx.nn.Linear(in_features=self.N_CHANNELS, 	# Updates agent velocity 
							         out_features=self.N_CHANNELS,
									 use_bias=False,
									 key=key6),
					   jax.nn.relu,									
					   eqx.nn.Linear(in_features=self.N_CHANNELS, 	# Updates agent velocity 
							         out_features=1,
									 use_bias=False,
									 key=key7),
						jax.nn.hard_tanh
					   ]
		# w_where = lambda l: l.weight
		# self.layers_pheremone[0] = eqx.tree_at(w_where,
 		# 									   self.layers_pheremone[0],
 		# 									   jax.random.uniform(key=key1,
 		# 														  shape=self.layers_pheremone[0].weight.shape,
 		# 														  minval=-0.01,
 		# 														  maxval=0.01)
 		# 									   )
		# self.layers_velocity_mag[0] = eqx.tree_at(w_where,
 		# 									   self.layers_velocity_mag[0],
 		# 									   jax.random.uniform(key=key2,
 		# 														  shape=self.layers_velocity_mag[0].weight.shape,
 		# 														  minval=-0.01,
 		# 														  maxval=0.01)
 		# 									   )
		# self.layers_d_angle[0] = eqx.tree_at(w_where,
 		# 									   self.layers_d_angle[0],
 		# 									   jax.random.uniform(key=key2,
 		# 														  shape=self.layers_d_angle[0].weight.shape,
 		# 														  minval=-0.01,
 		# 														  maxval=0.01)
 		# 									   )
	@eqx.filter_jit
	def __call__(self,X):
		for L in self.layers:
			X = L(X)
		v = X
		dp= X
		da= X
		for L in self.layers_pheremone:
			dp = L(dp)
		for L in self.layers_velocity_mag:
			v = L(v)
		for L in self.layers_d_angle:
			da = L(da)
		return v,dp,da
		
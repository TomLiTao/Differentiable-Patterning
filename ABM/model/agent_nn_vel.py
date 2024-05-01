import jax
import equinox as eqx
import jax.numpy as jnp
import time

class agent_nn(eqx.Module):
	layers: list
	layers_pheremone: list
	layers_velocity: list
	N_CHANNELS: int
	
	def __init__(self,N_CHANNELS,key=jax.random.PRNGKey(int(time.time()))):
		key1,key2,key3,key4 = jax.random.split(key,4)
		self.N_CHANNELS=N_CHANNELS
		def flat_func(x):
			return jnp.array(x).flatten()
		self.layers = [flat_func,
				       eqx.nn.Linear(in_features=3*self.N_CHANNELS,	# Reads local pheremone concentrations and combines them together
							         out_features=self.N_CHANNELS,
									 use_bias=True,
									 key=key1),
				       jax.nn.relu
					#    eqx.nn.Linear(in_features=self.N_CHANNELS,	# Reads local pheremone concentrations and combines them together
					# 		         out_features=self.N_CHANNELS,
					# 				 use_bias=True,
					# 				 key=key2),
					#    jax.nn.relu
					   ]
		self.layers_pheremone = [ 
					   eqx.nn.Linear(in_features=self.N_CHANNELS, 	# Updates the pheremone concentrations at agent position
							         out_features=self.N_CHANNELS,
									 use_bias=True,
									 key=key3),
					   jnp.tanh 									# Bounds updates of pheremone between 1 and -1
					   ]	
		self.layers_velocity = [									
					   eqx.nn.Linear(in_features=self.N_CHANNELS, 	# Updates agent velocity 
							         out_features=2,
									 use_bias=False,
									 key=key4),
					   jnp.tanh 									# Bounds agent velocity components between 1 and -1
					   ]
		w_where = lambda l: l.weight
		# self.layers_pheremone[0] = eqx.tree_at(w_where,
		# 									   self.layers_pheremone[0],
		# 									   jax.random.uniform(key=key3,
		# 														  shape=self.layers_pheremone[0].weight.shape,
		# 														  minval=-1,
		# 														  maxval=1)
		# 									   )
		# self.layers_velocity[0] = eqx.tree_at(w_where,
		# 									   self.layers_velocity[0],
		# 									   jax.random.uniform(key=key4,
		# 														  shape=self.layers_velocity[0].weight.shape,
		# 														  minval=-1,
		# 														  maxval=1)
		# 									   )
	@eqx.filter_jit
	def __call__(self,X):
		for L in self.layers:
			X = L(X)
		#print(X.shape)
		v = X
		dp= X
		for L in self.layers_pheremone:
			dp = L(dp)
		for L in self.layers_velocity:
			v = L(v)
		#print(v.shape)
		#print(dp.shape)
		return v,dp
		
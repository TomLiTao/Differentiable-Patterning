import jax
import time
import jax.numpy as np
import jax.random as jr
from jaxtyping import Float, Int, PyTree, Scalar, Array
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract


class DataAugmenter(DataAugmenterAbstract):
	"""
		Inherits the methods of DataAugmenter, but overwrites the batch cloning in the init
	"""
	def data_init(self,SHARDING=None):
		return None
	
	
	def sub_trajectory_split(self,L,key=jax.random.PRNGKey(int(time.time()))):
		"""
		Splits data into x (initial conditions) and y (following sub trajectory of length L).
		So that x[:,i]->y[:,i+1:i+1+L] is learned

		Parameters
		----------
		L : Int
			Length of subdirectory

		Returns
		-------
		x : float32[BATCHES,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : float32[BATCHES,L,CHANNELS,WIDTH,HEIGHT]
			Following sub trajectories

		"""
		
		
		pos = list(jax.random.randint(key,shape=(len(self.data_saved),),minval=0,maxval=self.data_true[0].shape[0]-L-1))
		x = jax.tree_util.tree_map(lambda data,p:data[p],self.data_saved,pos)
		y = jax.tree_util.tree_map(lambda data,p:data[p+1:p+1+L],self.data_saved,pos)
		
		return x,y
		
		
	def data_callback(self,
				   	  x: PyTree[Float[Array, "1 C W H"]], 
					  y: PyTree[Float[Array, "{L} C W H"]], 
					  i: Int[Scalar, ""],
					  L: Int[Scalar, ""],
					  key):
		
		""" Preserves intermediate hidden channels after each training iteration

		Returns:
			x : PyTree [BATCHES] f32[1,CHANNELS,WIDTH,HEIGHT]
				Initial conditions
			y : PyTree [BATCHES] f32[L,CHANNELS,WIDTH,HEIGHT]
				True trajectories that follow the initial conditions
		"""
		
		data = self.return_saved_data()
		B = len(data)
		N = data[0].shape[0]
		C = self.OBS_CHANNELS


		# y input contains important hidden channel information that we want to preserve during training
		if hasattr(self,"SUBTRAJECTORY_LOCATION"):
			pos = self.SUBTRAJECTORY_LOCATION
			preserve_hidden_channels = lambda data,y,p: data.at[p+1:p+1+L,C:].set(y[:,C:])
			data = jax.tree_util.tree_map(preserve_hidden_channels,data,y,pos)
			self.save_data(data)
		
		# Update position counters
		# with probability p, increment each counter by 1, with probability 1-p, reset each counter to a random value
		#  	If a counter reaches the end of the data, reset it to a random value
			
		reset_counters = jr.bernoulli(key,p=0.01,shape=(B,))
		_pos_incremented = np.clip(np.array(self.SUBTRAJECTORY_LOCATION)+1,min=0,max=N-L-1)
		_pos_at_end = _pos_incremented == N-L-1
		reset_counters = np.logical_or(reset_counters,_pos_at_end)
		#_pos_reset = jax.random.randint(key,shape=(B,),minval=0,maxval=N-L-1)
		_pos_reset = np.zeros(shape=(B,)).astype(int)
		pos = np.where(reset_counters,_pos_reset,_pos_incremented)
		pos = list(pos)

		# Sample new pairs of x and y
		x = jax.tree_util.tree_map(lambda data,p:data[p],self.data_saved,pos)
		y = jax.tree_util.tree_map(lambda data,p:data[p+1:p+1+L],self.data_saved,pos)
		self.SUBTRAJECTORY_LOCATION = pos

		return x,y

		
	def data_load(self,L,key):
		data = self.return_saved_data()
		pos = list(jax.random.randint(key,shape=(len(data),),minval=0,maxval=data[0].shape[0]-L-1))
		x0 = jax.tree_util.tree_map(lambda d,p:d[p],data,pos)
		y0 = jax.tree_util.tree_map(lambda d,p:d[p+1:p+1+L],data,pos)
		self.SUBTRAJECTORY_LOCATION = pos
		return x0,y0

		

import jax
import time
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
		
		
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_textures
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
import jax
import time

class DataAugmenterCustom(DataAugmenterAbstract):
	def data_init(self,SHARDING = None):
		"""
		Chain together various data augmentations to perform at intialisation of NCA training

		"""
		data = self.return_saved_data()
		data = self.duplicate_batches(data, 4)
		#data = self.pad(data, 10)

		
		self.save_data(data)
		return None

	def data_callback(self,x,y,i):
		"""
		Called after every training iteration to perform data augmentation and processing		


		Parameters
		----------
		x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Final states
		i : int
			Current training iteration - useful for scheduling mid-training data augmentation

		Returns
		-------
		x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Initial conditions
		y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
			Final states

		"""

		x_true,_ =self.split_x_y(1)
				
		propagate_xn = lambda x:x.at[1:].set(x[:-1])
		reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
		
		x = jax.tree_util.tree_map(propagate_xn,x) # Set initial condition at each X[n] at next iteration to be final state from X[n-1] of this iteration
		x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
		
		if i < 500:		
			for b in range(len(x)//2):
				x[b*2] = x[b*2].at[:,:self.OBS_CHANNELS].set(x_true[b*2][:,:self.OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
		
		if hasattr(self, "PREVIOUS_KEY"):
			key = jax.random.fold_in(self.PREVIOUS_KEY,i)
		else:
			key=jax.random.PRNGKey(int(time.time()))
		x = self.noise(x,0.005,key=key)
		self.PREVIOUS_KEY = key
		return x,y
		







CHANNELS=16

#data = load_emoji_sequence(["crab.png","alien_monster.png","alien_monster.png"],downsample=2)
data = load_textures(["dotted/dotted_0109.jpg","braided/braided_0075.jpg"],downsample=4,crop_square=True)
t=64
iters=2000



nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=True)
opt = NCA_Trainer(nca,
				  data,
				  #model_filename="micropattern_radii_sized_b"+str(B)+"_r1e-2_v2_"+str(index),
				  model_filename="emoji_texture_nca_test_6",
				  DATA_AUGMENTER=DataAugmenterCustom)
				 # BOUNDARY_MASK=masks,
				    

opt.train(t,iters,WARMUP=10)
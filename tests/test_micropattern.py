from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import *
from NCA.utils import *
from NCA.NCA_visualiser import *
import optax
import numpy as np
import jax.numpy as jnp
import jax

CHANNELS = 16
t = 64
iters=1000

data,masks = load_micropattern_radii("../Data/micropattern_radii/Chir_Fgf_*/processed/*")
print(data[0].shape)
print(masks[0].shape)
data = data[:2]
masks = masks[:2]


schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
optimiser= optax.adamw(schedule)
#optimiser= optax.chain(optax.clip_by_block_rms(1.0),optimiser)



class data_augmenter_subclass(DataAugmenter):
	 #Redefine how data is pre-processed before training
	 # Remove most of the data augmentation - don't need shifting or extra batches or intermediate propagation
	 def data_init(self,batches):
		  data = self.return_saved_data()
		  self.save_data(data)
		  return None  
	 def data_callback(self, x, y, i):
		 x_true,_ =self.split_x_y(1)
		 reset_x0 = lambda x,x_true:x.at[0].set(x_true[0])
		 x = jax.tree_util.tree_map(reset_x0,x,x_true) # Keep first initial x correct
		 return x,y
	
nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=False)


opt = NCA_Trainer(nca,
				  data,
				  model_filename="jax_micropattern_random_loss_test_3",
				  BOUNDARY_MASK=masks,
				  DATA_AUGMENTER = data_augmenter_subclass)

opt.train(t,iters,optimiser=optimiser,WARMUP=10)
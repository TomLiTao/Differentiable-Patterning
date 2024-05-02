#from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_smooth_model import cNCA
from NCA.model.NCA_smooth_gated_model import gcNCA
from NCA.model.NCA_gated_bounded_model import gbNCA
from NCA.model.NCA_bounded_model import bNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
import jax
from Common.utils import load_textures
#from Common.trainer.data_augmenter_tree_noise_ic import DataAugmenterNoise
#from Common.trainer.data_augmenter_tree_subsample_noise import DataAugmenterSubsampleNoiseTexture
from NCA.trainer.data_augmenter_nca import DataAugmenter
import time
import optax



CHANNELS=16
t=64
iters=2000


#data = load_textures(["dotted/dotted_0109.jpg","dotted/dotted_0109.jpg","honeycombed/honeycombed_0078.jpg","grid/grid_0002.jpg"],downsample=3,crop_square=True,crop_factor=1)
data = load_textures(["banded/banded_0109.jpg","dotted/dotted_0109.jpg","honeycombed/honeycombed_0078.jpg"],downsample=2,crop_square=True,crop_factor=1.5)
schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))

nca = bNCA(CHANNELS,OBSERVABLE_CHANNELS=4,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=True)
print(nca)

class da_subclass(DataAugmenter):
    def data_init(self, SHARDING=None):
        data = self.return_saved_data()
        data = self.duplicate_batches(data, 4)
        self.save_data(data)
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
        
        #if i < 500:		
        for b in range(len(x)//2):
            x[b*2] = x[b*2].at[:,:self.OBS_CHANNELS].set(x_true[b*2][:,:self.OBS_CHANNELS]) # Set every other batch of intermediate initial conditions to correct initial conditions
            
        
        if hasattr(self, "PREVIOUS_KEY"):
            key = jax.random.fold_in(self.PREVIOUS_KEY,i)
        else:
            key=jax.random.PRNGKey(int(time.time()))

        x = self.noise(x,0.005,key=key)
        self.PREVIOUS_KEY = key
        return x,y

opt = NCA_Trainer(nca,
				  data,
				  model_filename="texture_bounded_nca_test_1",
				  DATA_AUGMENTER=da_subclass)
				  
				    

opt.train(t,
          iters,
          WARMUP=10,
          optimiser=optimiser,
          LOSS_FUNC_STR="vgg",
          LOOP_AUTODIFF="checkpointed")
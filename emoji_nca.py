#from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_emoji_sequence
#from Common.trainer.data_augmenter_tree_noise_ic import DataAugmenterNoise
#from Common.trainer.data_augmenter_tree_subsample import DataAugmenterSubsampleNoiseTexture
from NCA.trainer.data_augmenter_nca import DataAugmenter
import time
import optax



CHANNELS=16
t=64
iters=2000

data = load_emoji_sequence(["crab.png","alien_monster.png","butterfly.png","butterfly.png"],downsample=2)
#data = load_textures(["dotted/dotted_0109.jpg","honeycombed/honeycombed_0078.jpg","grid/grid_0002.jpg"],downsample=3,crop_square=True,crop_factor=1)
schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.adamw(schedule))

nca = gNCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=True)
print(nca)
opt = NCA_Trainer(nca,
				  data,
				  model_filename="gate_emoji_nca_test_1",
				  DATA_AUGMENTER=DataAugmenter)
				  
				    

opt.train(t,
          iters,
          WARMUP=10,
          optimiser=optimiser,
          LOSS_FUNC_STR="euclidean")
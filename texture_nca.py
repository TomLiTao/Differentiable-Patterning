#from NCA.model.NCA_model import NCA
#from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_smooth_gated_model import gcNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_textures
#from Common.trainer.data_augmenter_tree_noise_ic import DataAugmenterNoise
from Common.trainer.data_augmenter_tree_subsample_noise import DataAugmenterSubsampleNoiseTexture
import time
import optax



CHANNELS=16
t=64
iters=2000


#data = load_textures(["dotted/dotted_0109.jpg","dotted/dotted_0109.jpg","honeycombed/honeycombed_0078.jpg","grid/grid_0002.jpg"],downsample=3,crop_square=True,crop_factor=1)
data = load_textures(["banded/banded_0109.jpg","banded/banded_0109.jpg","perforated/perforated_0106.jpg","perforated/perforated_0106.jpg"],downsample=1,crop_square=True,crop_factor=1)
schedule = optax.exponential_decay(4e-2, transition_steps=iters, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.adamw(schedule))

nca = gcNCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],KERNEL_SCALE=3,FIRE_RATE=0.5,PERIODIC=True)
print(nca)

class da_subclass(DataAugmenterSubsampleNoiseTexture):
    def __init__(self, data_true, hidden_channels=0):
        super().__init__(data_true, hidden_channels)
        self.sample_size = 64
        self.resample_freq = 8

opt = NCA_Trainer(nca,
				  data,
				  model_filename="texture_smooth_gated_highres_nca_test_1",
				  DATA_AUGMENTER=da_subclass)
				  
				    

opt.train(t,
          iters,
          WARMUP=10,
          optimiser=optimiser,
          LOSS_FUNC_STR="vgg")
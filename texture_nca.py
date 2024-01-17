from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_textures
#from Common.trainer.data_augmenter_tree_noise_ic import DataAugmenterNoise
from Common.trainer.data_augmenter_tree_subsample import DataAugmenterSubsampleNoiseTexture
import time


CHANNELS=16

#data = load_emoji_sequence(["crab.png","alien_monster.png","alien_monster.png"],downsample=2)
data = load_textures(["dotted/dotted_0109.jpg","honeycombed/honeycombed_0059.jpg","veined/veined_0095.jpg"],downsample=1,crop_square=True,crop_factor=1)
t=64
iters=2000



nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=True)
opt = NCA_Trainer(nca,
				  data,
				  #model_filename="micropattern_radii_sized_b"+str(B)+"_r1e-2_v2_"+str(index),
				  model_filename="emoji_texture_nca_test_18",
				  DATA_AUGMENTER=DataAugmenterSubsampleNoiseTexture)
				  #BOUNDARY_MASK=masks,
				    

opt.train(t,iters,WARMUP=10)
from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.utils import load_emoji_sequence

CHANNELS=16

data = load_emoji_sequence(["crab.png","alien_monster.png","alien_monster.png"],downsample=2)
t=64
iters=500



nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=0.5,PERIODIC=False)
opt = NCA_Trainer(nca,
				  data,
				  #model_filename="micropattern_radii_sized_b"+str(B)+"_r1e-2_v2_"+str(index),
				  model_filename="emoji_texture_nca_test")
				 # BOUNDARY_MASK=masks,
				    

opt.train(t,iters,WARMUP=10)
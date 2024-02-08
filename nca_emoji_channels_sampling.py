from NCA.model.NCA_model import NCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_emoji_sequence
from Common.eddie_indexer import index_to_channel_sample
#from Common.trainer.data_augmenter_tree_noise_ic import DataAugmenterNoise
#from Common.trainer.data_augmenter_tree_subsample import DataAugmenterSubsampleNoiseTexture
import time
import jax.random as jr
import optax
import sys



index=int(sys.argv[1])-1
key = jr.PRNGKey(int(time.time()))
key = jr.fold_in(key,index)
key_model,key_trainer = jr.split(key,2)
N_BATCHES = 4
TRAIN_ITERS = 8000
LEARN_RATE = 1e-3
N_CHANNELS,SAMPLING = index_to_channel_sample(index)
FILENAME = "model_exploration/emoji_"+str(N_CHANNELS)+"_channels_"+str(SAMPLING)+"_sampling_v1"


schedule = optax.exponential_decay(1e-2, transition_steps=TRAIN_ITERS, decay_rate=0.99)
optimiser = optax.adamw(schedule)

data = load_emoji_sequence(["alien_monster.png","microbe.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)

nca = NCA(N_CHANNELS=N_CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],PERIODIC=False,key=key_model)
opt = NCA_Trainer(nca,data,model_filename=FILENAME)
opt.train(SAMPLING,TRAIN_ITERS,optimiser=optimiser,key=key_trainer)
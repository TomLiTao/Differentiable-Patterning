import jax
#jax.config.update("jax_enable_x64", True)
# import os
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True '
# )

from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from NCA.model.NCA_KAN_model import kaNCA
from NCA.trainer.NCA_trainer import NCA_Trainer
from Common.utils import load_emoji_sequence
from Common.eddie_indexer import index_to_data_nca_type
from NCA.trainer.data_augmenter_nca import DataAugmenter
import time
import optax
import sys

index=int(sys.argv[1])-1
data_index,nca_type_index = index_to_data_nca_type(index)
print(data_index)
print(nca_type_index)
CHANNELS=32
DOWNSAMPLE = 1
t=64
iters=8000

key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

class data_augmenter_subclass(DataAugmenter):
    #Redefine how data is pre-processed before training
    def data_init(self,SHARDING=None):
        data = self.return_saved_data()
        data = self.duplicate_batches(data, 4)
        data = self.pad(data, 10) 		
        self.save_data(data)
        return None

if data_index == 0:
    data = load_emoji_sequence(["crab.png","microbe.png","alien_monster.png","alien_monster.png"],downsample=DOWNSAMPLE)
    data_filename = "cr_mi_al"
if data_index == 1:
    data = load_emoji_sequence(["microbe.png","avocado_1f951.png","alien_monster.png","alien_monster.png"],downsample=DOWNSAMPLE)
    data_filename = "mi_av_al"
if data_index == 2:
    data = load_emoji_sequence(["avocado_1f951.png","mushroom_1f344.png","lizard_1f98e.png","lizard_1f98e.png"],downsample=DOWNSAMPLE)
    data_filename = "av_mu_li"
if data_index == 3:
    data = load_emoji_sequence(["mushroom_1f344.png","alien_monster.png","rooster_1f413.png","rooster_1f413.png"],downsample=DOWNSAMPLE)
    data_filename = "mu_al_ro"
# if data_index == 4:
#     data = load_emoji_sequence(["alien_monster.png","mushroom_1f344.png","alien_monster.png","alien_monster.png"],downsample=DOWNSAMPLE)
#     data_filename = "al_mu_al"
# if data_index == 5:
#     data = load_emoji_sequence(["butterfly.png","eye.png","lizard_1f98e.png","lizard_1f98e.png"],downsample=DOWNSAMPLE)
#     data_filename = "bu_ey_li"
# if data_index == 6:
#     data = load_emoji_sequence(["rooster_1f413.png","butterfly.png","eye.png","eye.png"],downsample=DOWNSAMPLE)
#     data_filename = "ro_bu_ey"
# if data_index == 7:
#     data = load_emoji_sequence(["lizard_1f98e.png","microbe.png","anatomical_heart.png","anatomical_heart.png"],downsample=DOWNSAMPLE)
#     data_filename = "li_mi_an"

# print(data_filename)
# print(nca_type_index)

schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))

if nca_type_index==0:
    print("Training anisotropic nca")
    nca = NCA(CHANNELS,
              KERNEL_STR=["ID","LAP","GRAD"],
              KERNEL_SCALE=1,
              FIRE_RATE=0.5,
              PADDING="REPLICATE",
              key=key)
    opt = NCA_Trainer(nca,
                      data,
                      model_filename="demo_stable_emoji_anisotropic_nca_"+data_filename,
                      DATA_AUGMENTER=data_augmenter_subclass,
                      GRAD_LOSS=True)

    opt.train(t,
        iters,
        WARMUP=10,
        optimiser=optimiser,
        LOSS_FUNC_STR="euclidean",
        key=key)

if nca_type_index==1: 
    print("Training anisotropic gated nca")
    nca = gNCA(CHANNELS,
               KERNEL_STR=["ID","LAP","GRAD"],
               KERNEL_SCALE=1,
               FIRE_RATE=0.5,
               PADDING="REPLICATE",
               key=key)
    opt = NCA_Trainer(nca,
                      data,
                      model_filename="demo_stable_emoji_anisotropic_gated_nca_"+data_filename,
                      DATA_AUGMENTER=data_augmenter_subclass,
                      GRAD_LOSS=True)
                    
    opt.train(t,
            iters,
            WARMUP=10,
            optimiser=optimiser,
            LOSS_FUNC_STR="euclidean",
            key=key)


if nca_type_index==2:
    print("Training isotropic nca")
    nca = NCA(CHANNELS,
              KERNEL_STR=["ID","LAP","DIFF"],
              KERNEL_SCALE=1,
              FIRE_RATE=0.5,
              PADDING="REPLICATE",
              key=key)
    opt = NCA_Trainer(nca,
                      data,
                      model_filename="demo_stable_emoji_isotropic_nca_"+data_filename,
                      DATA_AUGMENTER=data_augmenter_subclass,
                      GRAD_LOSS=True)
                
    opt.train(t,
            iters,
            WARMUP=10,
            optimiser=optimiser,
            LOSS_FUNC_STR="euclidean",
            key=key)

if nca_type_index==3: 
    print("Training isotropic gated nca")
    nca = gNCA(CHANNELS,
               KERNEL_STR=["ID","LAP","DIFF"],
               KERNEL_SCALE=1,
               FIRE_RATE=0.5,
               PADDING="REPLICATE",
               key=key)
    opt = NCA_Trainer(nca,
                      data,
                      model_filename="demo_stable_emoji_isotropic_gated_nca_"+data_filename,
                      DATA_AUGMENTER=data_augmenter_subclass,
                      GRAD_LOSS=True)
            
    opt.train(t,
            iters,
            WARMUP=10,
            optimiser=optimiser,
            LOSS_FUNC_STR="euclidean",
            key=key)




				  	    

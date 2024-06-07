import jax
import jax.numpy as np
import optax
from PDE.trainer.optimiser import non_negative_diffusion_chemotaxis
from einops import repeat
from PDE.model.reaction_diffusion_chemotaxis.update import F
from PDE.model.solver.semidiscrete_solver import PDE_solver
from PDE.trainer.PDE_trajectory_trainer import PDE_Trainer
from NCA.model.NCA_model import NCA
from NCA.model.NCA_gated_model import gNCA
from Common.utils import load_emoji_sequence
from Common.eddie_indexer import index_to_pde_hyperparameters
from NCA.trainer.data_augmenter_nca import DataAugmenter as DataAugmenterNCA
import time
import sys




CHANNELS = 32
CELL_CHANNELS = 3
SIGNAL_CHANNELS = CHANNELS-CELL_CHANNELS
DOWNSAMPLE=3
index=int(sys.argv[1])-1




PARAMS = index_to_pde_hyperparameters(index)
INTERNAL_ACTIVATIONS = PARAMS[0]
OUTER_ACTIVATIONS = PARAMS[1]
INIT_SCALE = PARAMS[2]
STABILITY_FACTOR = PARAMS[3]
OPTIMISER = PARAMS[4]
LEARN_RATE = PARAMS[5]
TRAJECTORY_LENGTH = PARAMS[6]
USE_BIAS = PARAMS[7]
INTERNAL_TEXT = PARAMS[8]
OUTER_TEXT = PARAMS[9]
OPTIMISER_TEXT = PARAMS[10]
LEARN_RATE_TEXT = PARAMS[11]

key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)
# Load saved NCA model and data, then run to generate trajectory to train PDE to



data = load_emoji_sequence(["crab.png","microbe.png","alien_monster.png"],downsample=DOWNSAMPLE)
data_filename = "cr_mi_al"
da = DataAugmenterNCA(data,28)
da.data_init()
x0 = np.array(da.split_x_y()[0])[0,0]



nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","GRAD"],KERNEL_SCALE=1,PADDING="REPLICATE")
nca = nca.load("models/demo_lowres_stable_emoji_anisotropic_nca_"+data_filename+".eqx")
nca_type="_pure_anisotropic_"

NCA_trajectory = nca.run(96,x0)
NCA_trajectory = repeat(NCA_trajectory,"T C X Y -> B T C X Y",B=4)
print(NCA_trajectory.shape)


# Define PDE model
func = F(CELL_CHANNELS,
         SIGNAL_CHANNELS,
         PADDING="REPLICATE",
         dx=1.0,
         INTERNAL_ACTIVATION=INTERNAL_ACTIVATIONS,
         OUTER_ACTIVATION=OUTER_ACTIVATIONS,
         INIT_SCALE=INIT_SCALE,
         STABILITY_FACTOR=STABILITY_FACTOR,
         USE_BIAS=USE_BIAS,
         key=key)
pde = PDE_solver(func,dt=0.1)


# Define optimiser and lr schedule
iters = 2000
schedule = optax.exponential_decay(LEARN_RATE, transition_steps=iters, decay_rate=0.99)
opt = non_negative_diffusion_chemotaxis(schedule,optimiser=OPTIMISER)



trainer = PDE_Trainer(pde,
                      NCA_trajectory,
                      #model_filename="pde_hyperparameters_chemreacdiff_emoji_anisotropic_nca_2/init_scale_"+str(INIT_SCALE)+"_stability_factor_"+str(STABILITY_FACTOR)+"act_"+INTERNAL_TEXT+"_"+OUTER_TEXT)
                      model_filename="pde_hyperparameters_chemreacdiff_emoji_anisotropic_nca_2/act_"+INTERNAL_TEXT+"_"+OUTER_TEXT+"_opt_"+OPTIMISER_TEXT+"_lr_"+LEARN_RATE_TEXT+"_tl_"+str(TRAJECTORY_LENGTH)+"_bias_"+str(USE_BIAS))
trainer.train(TRAJECTORY_LENGTH,iters,optimiser=opt,LOG_EVERY=100)
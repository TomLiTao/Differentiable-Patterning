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




INTERNAL_ACTIVATIONS,OUTER_ACTIVATIONS,INIT_SCALE,STABILITY_FACTOR,INTERNAL_TEXT,OUTER_TEXT = index_to_pde_hyperparameters(index)
key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)
# Load saved NCA model and data, then run to generate trajectory to train PDE to



data = load_emoji_sequence(["crab.png","microbe.png","alien_monster.png","alien_monster.png"],downsample=DOWNSAMPLE)
data_filename = "cr_mi_al"
#data = load_emoji_sequence(["alien_monster.png","microbe.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)
da = DataAugmenterNCA(data,28)
da.data_init()
x0 = np.array(da.split_x_y()[0])[0,0]



#nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","DIFF"],FIRE_RATE=1.0,PERIODIC=False)
#nca = nca.load("models/model_exploration/emoji_16_channels_64_sampling_relu_activation_ID_DIFF_LAP_kernels_v4.eqx")







nca = NCA(CHANNELS,KERNEL_STR=["ID","LAP","GRAD"],KERNEL_SCALE=1,PADDING="REPLICATE")
nca = nca.load("models/demo_lowres_stable_emoji_anisotropic_nca_"+data_filename+".eqx")
nca_type="_pure_anisotropic_"

NCA_trajectory = nca.run(128,x0)
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
         key=key)
pde = PDE_solver(func,dt=0.1)


# Define optimiser and lr schedule
iters = 1000
schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
opt = non_negative_diffusion_chemotaxis(schedule)



trainer = PDE_Trainer(pde,
                      NCA_trajectory,
                      model_filename="pde_hyperparameters_chemreacdiff_emoji_anisotropic_nca/init_scale_"+str(INIT_SCALE)+"_stability_factor_"+str(STABILITY_FACTOR)+"act_"+INTERNAL_TEXT+"_"+OUTER_TEXT)

trainer.train(32,iters,optimiser=opt)
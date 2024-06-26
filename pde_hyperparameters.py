import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import optax
from PDE.trainer.optimiser import non_negative_diffusion_chemotaxis
from einops import repeat
from PDE.model.reaction_diffusion_chemotaxis.update import F
from PDE.model.solver.semidiscrete_solver import PDE_solver
from PDE.trainer.PDE_trajectory_trainer import PDE_Trainer
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
from PDE.model.fixed_models.update_chhabra import F as F_chhabra
from PDE.model.fixed_models.update_hillen_painter import F as F_hillen_painter
from PDE.model.fixed_models.update_cahn_hilliard import F as F_cahn_hilliard
from Common.eddie_indexer import index_to_pde_hyperparameters
from NCA.trainer.data_augmenter_nca import DataAugmenter as DataAugmenterNCA
from Common.model.spatial_operators import Ops
from einops import rearrange
import time
import sys


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
EQUATION_INDEX = PARAMS[12]

key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)
# Load saved NCA model and data, then run to generate trajectory to train PDE to



CHANNELS = 16
CELL_CHANNELS = 1
SIGNAL_CHANNELS = CHANNELS-CELL_CHANNELS
SIZE = 64
BATCHES = 4



# data = load_emoji_sequence(["microbe.png","avocado_1f951.png","alien_monster.png"],downsample=DOWNSAMPLE)
# data_filename = "mi_av_al"
# da = DataAugmenterNCA(data,28)
if EQUATION_INDEX==0:
    PDE_STR = "gray_scott"
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))
    op = Ops(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=3)
    v_av = jax.vmap(op.Average,in_axes=0,out_axes=0)
    for i in range(5):
        x0 = v_av(x0)
    x0 = x0.at[:,0].set(jnp.where(x0[:,0]>0.51,1.0,0.0))
    x0 = x0.at[:,1].set(1-x0[:,0])
    func = F_gray_scott(PADDING="REFLECT",dx=1.0,KERNEL_SCALE=1)
    v_func = jax.vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.1)
    T,Y = solver(ts=jnp.linspace(0,10000,101),y0=x0)

elif EQUATION_INDEX==1:
    PDE_STR = "chhabra"
    scale=0.5
    x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))*scale
    func = F_chhabra(PADDING="CIRCULAR",dx=0.5,KERNEL_SCALE=1)
    v_func = jax.vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.1)
    T,Y = solver(ts=jnp.linspace(0,5000,101),y0=x0)

elif EQUATION_INDEX==2:
    PDE_STR = "hillen_painter"
    scale=0.1  
    x0 = jnp.ones((BATCHES,2,SIZE,SIZE))
    x0 = x0.at[:,1].set((1-scale)*x0[:,0]+jr.uniform(key,shape=(BATCHES,SIZE,SIZE))*scale)
    func = F_hillen_painter(PADDING="CIRCULAR",
                            dx=0.25,
                            KERNEL_SCALE=1)
    v_func = jax.vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.01)
    T,Y = solver(ts=jnp.linspace(0,100,101),y0=x0)
elif EQUATION_INDEX==3:
    PDE_STR = "cahn_hilliard"
    scale=2.0
    x0 = jr.uniform(key,shape=(BATCHES,1,SIZE,SIZE))*scale - 1
    func = F_cahn_hilliard(PADDING="CIRCULAR",dx=1.5,KERNEL_SCALE=1)
    v_func = jax.vmap(func,in_axes=(None,0,None),out_axes=0)
    solver = PDE_solver(v_func,dt=0.5)
    T,Y = solver(ts=jnp.linspace(0,20000,101),y0=x0)

Y = rearrange(Y,"T B C X Y -> B T C X Y")
Y = Y[:,:,:1] # Only include main channel, not inhibitor/other chemical
Y = jnp.pad(Y,((0,0),(0,0),(0,CHANNELS-1),(0,0),(0,0)),mode="constant")
print(Y.shape)


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
                      Y,
                      #model_filename="pde_hyperparameters_chemreacdiff_emoji_anisotropic_nca_2/init_scale_"+str(INIT_SCALE)+"_stability_factor_"+str(STABILITY_FACTOR)+"act_"+INTERNAL_TEXT+"_"+OUTER_TEXT)
                      model_filename="pde_hyperparameters_chemreacdiff/"+PDE_STR+"_act_"+INTERNAL_TEXT+"_"+OUTER_TEXT+"_opt_"+OPTIMISER_TEXT+"_lr_"+LEARN_RATE_TEXT+"_tl_"+str(TRAJECTORY_LENGTH)+"_bias_"+str(USE_BIAS))
trainer.train(TRAJECTORY_LENGTH,iters,optimiser=opt,LOG_EVERY=100)
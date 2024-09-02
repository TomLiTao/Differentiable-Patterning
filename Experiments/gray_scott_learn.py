import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import jax.random as jr
import equinox as eqx
import optax
from PDE.trainer.optimiser import non_negative_diffusion
from PDE.trainer.optimiser import multi_learnrate
from einops import repeat
from PDE.model.reaction_diffusion_advection.update import F
from PDE.model.solver.semidiscrete_solver import PDE_solver
from PDE.trainer.PDE_trainer import PDE_Trainer
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
# from PDE.model.fixed_models.update_chhabra import F as F_chhabra
# from PDE.model.fixed_models.update_hillen_painter import F as F_hillen_painter
# from PDE.model.fixed_models.update_cahn_hilliard import F as F_cahn_hilliard
from Common.eddie_indexer import index_to_pde_gray_scott_hyperparameters
from Common.model.spatial_operators import Ops
from einops import rearrange
import time
import sys

index=int(sys.argv[1])-1


PARAMS = index_to_pde_gray_scott_hyperparameters(index)
# INIT_SCALE = {"reaction":0.1,"advection":0.3,"diffusion":0.3}
STABILITY_FACTOR = 0.01


key = jax.random.PRNGKey(int(time.time()))
key = jax.random.fold_in(key,index)

CHANNELS = 16
ITERS = 1001
SIZE = 64
BATCHES = 8
PADDING = "CIRCULAR"
TRAJECTORY_LENGTH = PARAMS["TRAJECTORY_LENGTH"]
PDE_STR = "gray_scott"
dt = 1.0

pde_hyperparameters = {"N_CHANNELS":CHANNELS,
                       "PADDING":PADDING,
                       "INTERNAL_ACTIVATION":PARAMS["INTERNAL_ACTIVATIONS"],
                       "dx":1.0,
                       "TERMS":["reaction","diffusion"],
                       "ADVECTION_OUTER_ACTIVATION":"tanh",
                       "INIT_SCALE":{"reaction":0.1,"diffusion":0.1},
                       "INIT_TYPE":{"reaction":PARAMS["REACTION_INIT"],"diffusion":PARAMS["DIFFUSION_INIT"]},
                       "STABILITY_FACTOR":0.01,
                       "USE_BIAS":True,
                       "ORDER":PARAMS["ORDER"],
                       "N_LAYERS":PARAMS["N_LAYERS"],
                       "ZERO_INIT":{"reaction":False,"diffusion":False}}
solver_hyperparameters = {"dt":dt,
                          "SOLVER":"euler",
                          "rtol":1e-3,
                          "atol":1e-3,
                          "ADAPTIVE":False}
hyperparameters = {"pde":pde_hyperparameters,
                   "solver":solver_hyperparameters}


# ----------------- Define data -----------------

x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))
op = Ops(PADDING=PADDING,dx=1.0,KERNEL_SCALE=2)
v_av = eqx.filter_vmap(op.Average,in_axes=0,out_axes=0)
for i in range(1):
    x0 = v_av(x0)
x0 = x0.at[:,1].set(np.where(x0[:,1]>0.55,1.0,0.0))

x0 = x0.at[:,1,:SIZE//4].set(0)
x0 = x0.at[:,1,:,:SIZE//4].set(0)
x0 = x0.at[:,1,-SIZE//4:].set(0)
x0 = x0.at[:,1,:,-SIZE//4:].set(0)
for i in range(1):
    x0 = v_av(x0)
x0 = x0.at[:,0].set(1-x0[:,1])

func = F_gray_scott(PADDING=PADDING,dx=1.0,KERNEL_SCALE=1)
v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0)
solver = PDE_solver(v_func,dt=dt)
T,Y = solver(ts=np.linspace(0,2000,PARAMS["TIME_RESOLUTION"]),y0=x0)
Y = rearrange(Y,"T B C X Y -> B T C X Y")
#Y = Y[:,:,:1] # Only include main channel, not inhibitor/other chemical

#Y = 2*(Y-np.min(Y,axis=2,keepdims=True))/(np.max(Y,axis=2,keepdims=True)-np.min(Y,axis=2,keepdims=True)) - 1
Y = Y.at[:,:,0].set(2*(Y[:,:,0]-np.min(Y[:,:,0]))/(np.max(Y[:,:,0])-np.min(Y[:,:,0])) - 1)
Y = Y.at[:,:,1].set(2*(Y[:,:,1]-np.min(Y[:,:,1]))/(np.max(Y[:,:,1])-np.min(Y[:,:,1])) - 1)

# ----------------- Define model -----------------
func = F(key=key,**hyperparameters["pde"])
pde = PDE_solver(func,**hyperparameters["solver"])

# Define optimiser and lr schedule
#iters = 2000
#schedule = optax.exponential_decay(PARAMS["LEARN_RATE"], transition_steps=iters, decay_rate=0.99)
#opt = non_negative_diffusion(schedule,optimiser=OPTIMISER)
#opt = optax.chain(optax.scale_by_param_block_norm(),
			#PARAMS["OPTIMISER"](schedule))
schedule = optax.exponential_decay(1e-3, transition_steps=ITERS, decay_rate=0.99)
opt = multi_learnrate(
    schedule,
    rate_ratios={"advection": PARAMS["ADVECTION_RATIO"],
                 "reaction": PARAMS["REACTION_RATIO"],
                 "diffusion": 1},
    optimiser=PARAMS["OPTIMISER"],
    pre_process=PARAMS["OPTIMISER_PRE_PROCESS"],
)

trainer = PDE_Trainer(PDE_solver=pde,
                      PDE_HYPERPARAMETERS=hyperparameters,
                      data=Y,
                      Ts=T,
                      #model_filename="pde_hyperparameters_chemreacdiff_emoji_anisotropic_nca_2/init_scale_"+str(INIT_SCALE)+"_stability_factor_"+str(STABILITY_FACTOR)+"act_"+INTERNAL_TEXT+"_"+OUTER_TEXT)
                      model_filename="pde_hyperparameters_reacdiff_gray_scott_euler/lr_1e-3_ch_"+str(CHANNELS)+"_tl_"+str(PARAMS["TRAJECTORY_LENGTH"])+"_resolution_"+str(PARAMS["TIME_RESOLUTION"])+"_ord_"+str(PARAMS["ORDER"])+"_layers_"+str(PARAMS["N_LAYERS"])+"_act_"+PARAMS["INTERNAL_ACTIVATIONS_TEXT"]+"_R_"+PARAMS["REACTION_INIT"]+PARAMS["REACTION_ZERO_INIT_TEXT"]+"_lrr_"+PARAMS["REACTION_RATIO_TEXT"]+"_D_"+PARAMS["DIFFUSION_INIT"]+PARAMS["DIFFUSION_ZERO_INIT_TEXT"]+"_opt_"+PARAMS["OPTIMISER_TEXT"])

UPDATE_X0_PARAMS = {"iters":16,
                    "update_every":10000,
                    "optimiser":optax.nadam,
                    "learn_rate":1e-4,
                    "verbose":True}

trainer.train(SUBTRAJECTORY_LENGTH=TRAJECTORY_LENGTH,
              TRAINING_ITERATIONS=ITERS,
              OPTIMISER=opt,
              LOG_EVERY=50,
              WARMUP=1,
              LOSS_TIME_SAMPLING=PARAMS["LOSS_TIME_SAMPLING"],
              UPDATE_X0_PARAMS=UPDATE_X0_PARAMS)
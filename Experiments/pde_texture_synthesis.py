import jax
import jax.numpy as np
import jax.random as jr
import jax.tree as jt
import equinox as eqx
import time
import optax
from PDE.trainer.optimiser import multi_learnrate
from einops import rearrange
from PDE.model.reaction_diffusion_advection.update import F
from PDE.model.solver.semidiscrete_solver import PDE_solver
from PDE.trainer.PDE_texture_trainer import PDE_Trainer
from Common.utils import load_textures
import matplotlib.pyplot as plt
from Common.eddie_indexer import index_to_pde_texture_hyperparameters
from einops import rearrange,repeat
import sys
index=int(sys.argv[1])-1

PARAMS = index_to_pde_texture_hyperparameters(index)

CHANNELS = 8
ITERS = 1001
TRAJECTORY_LENGTH = 64
TRAJECTORY_SAMPLING = 1
WARMUP_WINDOW = 48
PADDING = "CIRCULAR"
LEARN_RATE = 5e-4
BATCHES = 8


INIT_SCALE = {"reaction":0.1,"advection":0.1,"diffusion":1.0}
INIT_TYPE = {"reaction":PARAMS["REACTION_INIT"],"advection":PARAMS["ADVECTION_INIT"],"diffusion":PARAMS["DIFFUSION_INIT"]}
ZERO_INIT = {"reaction":False,"advection":False,"diffusion":False}
key = jr.PRNGKey(int(time.time()))
key = jr.fold_in(key,index)
func = F(CHANNELS,
         PADDING=PADDING,
         dx=1.0,
         INTERNAL_ACTIVATION=jax.nn.tanh,
         ADVECTION_OUTER_ACTIVATION=jax.nn.tanh,
         INIT_SCALE=INIT_SCALE,
         INIT_TYPE=INIT_TYPE,
         STABILITY_FACTOR=0.001,
         USE_BIAS=True,
         ORDER=2,
         N_LAYERS=PARAMS["N_LAYERS"],
         ZERO_INIT=ZERO_INIT,
         key=key)
pde = PDE_solver(func,dt=0.1)



data = load_textures([PARAMS["FILENAME"]],downsample=3,crop_square=True,crop_factor=2)
data = repeat(data,"b t h w c -> b (t T) h w c",T=TRAJECTORY_LENGTH)
data = repeat(data,"b t h w c -> (b B) t h w c",B=BATCHES)
print(f"Data shape: {data.shape}")



schedule = optax.exponential_decay(LEARN_RATE, transition_steps=ITERS, decay_rate=0.99)

opt = multi_learnrate(
    schedule,
    rate_ratios={"advection": 1,
                 "reaction": 1,
                 "diffusion": 1},
    optimiser=optax.nadam,
    pre_process=PARAMS["OPTIMISER_PRE_PROCESS"],
)

trainer = PDE_Trainer(pde,
                      data,
                      model_filename="pde_textures/perlin_"+PARAMS["FILENAME_SHORT"]+"_nadam"+PARAMS["OPTIMISER_PRE_PROCESS_TEXT"]+"_ord_2_layers_"+str(PARAMS["N_LAYERS"])+"_A_"+PARAMS["ADVECTION_INIT"]+"_R_"+PARAMS["REACTION_INIT"]+"_D_"+PARAMS["DIFFUSION_INIT"])

trainer.train(t=TRAJECTORY_LENGTH,
              iters=ITERS,
              optimiser=opt,
              LOSS_TIME_WINDOW=WARMUP_WINDOW,
              LOSS_TIME_SAMPLING=TRAJECTORY_SAMPLING,
              LOG_EVERY=100,
              WARMUP=32)
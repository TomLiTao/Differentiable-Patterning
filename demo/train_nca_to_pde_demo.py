import jax
import jax.random as jr
import jax.numpy as jnp
import time
import optax
import equinox as eqx
import sys
sys.path.append('..')
from einops import rearrange
from Common.model.spatial_operators import Ops
from PDE.model.fixed_models.update_chhabra import F as F_chhabra
from PDE.model.solver.semidiscrete_solver import PDE_solver
from NCA.trainer.NCA_trainer import NCA_Trainer
from NCA.trainer.data_augmenter_nca_from_pde_2 import DataAugmenter
from NCA.model.NCA_model import NCA



# Set model and training parameters
ITERS = 8000        # Training iterations
CHANNELS = 8        # NCA channels
SIZE = 32           # Grid size
BATCHES = 2         # Data batches
TIME_SAMPLING = 32  # Timesteps between data snapshots

# Set up true PDE trajectory
key = jax.random.PRNGKey(int(time.time()))
scale=0.5
x0 = jr.uniform(key,shape=(BATCHES,2,SIZE,SIZE))*scale
func = F_chhabra(PADDING="CIRCULAR",dx=0.5,KERNEL_SCALE=1)
v_func = eqx.filter_vmap(func,in_axes=(None,0,None),out_axes=0) # Parallelise func over BATCHES axis
solver = PDE_solver(v_func,dt=0.1)
T,Y = solver(ts=jnp.linspace(0,5000,TIME_SAMPLING*8),y0=x0)
Y = rearrange(Y,"T B C X Y -> B T C X Y")                       # Reshape data so batch axis is first
Y = Y[:,:,:1]                                                   # Only include main channel, not inhibitor/other chemical
Y = 2*(Y-jnp.min(Y))/(jnp.max(Y)-jnp.min(Y)) - 1                # Rescale data between -1 and 1
Y = Y[:,::TIME_SAMPLING]                                        # Downsample along time axis



# Define NCA model
nca = NCA(N_CHANNELS=CHANNELS,
          KERNEL_STR=["ID","LAP"])

# Define NCA trainer
opt = NCA_Trainer(nca,
                  Y,
                  model_filename="demo/nca_pde_chhabra_example",
                  DATA_AUGMENTER=DataAugmenter,
                  GRAD_LOSS=True)

# Define optax.GradientTransformation() object to handle training
schedule = optax.exponential_decay(1e-4, transition_steps=ITERS, decay_rate=0.99)
optimiser = optax.chain(optax.scale_by_param_block_norm(),
                        optax.nadam(schedule))

# Do the training
opt.train(TIME_SAMPLING,
          ITERS,
          WARMUP=50,
          optimiser=optimiser,
          LOSS_FUNC_STR="euclidean",
          LOOP_AUTODIFF="lax",
          LOG_EVERY=50,
          key=key)
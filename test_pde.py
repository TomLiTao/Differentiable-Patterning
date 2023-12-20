import jax
import equinox as eqx
import jax.numpy as jnp
import time
import diffrax
import matplotlib.pyplot as plt
from PDE.reaction_diffusion_advection.advection import V
from PDE.reaction_diffusion_advection.reaction import R
from PDE.reaction_diffusion_advection.diffusion import D
from PDE.reaction_diffusion_advection.update import F
from PDE.solver.semidiscrete_solver import PDE_solver
from NCA.NCA_visualiser import my_animate
from PDE.trainer.optimiser import non_negative_diffusion
CHANNELS = 5
key=jax.random.PRNGKey(int(time.time()))
X = jnp.zeros((CHANNELS,32,32))
X = X.at[:,12:24,12:24].set(1.0)

#func = F(CHANNELS,True,key=key)

solver = PDE_solver(CHANNELS, False,dx=1,dt=0.01)
diff,static = solver.partition()
#pde = F(CHANNELS,False,dx=0.1)
optimiser = non_negative_diffusion(diff,1e-2,100)
print(optimiser)

opt_state = optimiser.init(diff)

#print(solver.func(0,X,args=None).shape)
#print(diffrax.ODETerm(solver.func).vf(0,X,args=None).shape)
#print(solver)
#print(update(X))
#diff,static = solver.partition()
#solver.combine(diff, static)
#print(solver)
#print(diff)
#T,Y = solver(ts=jnp.linspace(0,100,101),y0=X)
#my_animate(Y[:,:3])
#plt.imshow(Y[-1,0])
#plt.show()
#print(T)


#print(update.f_v)
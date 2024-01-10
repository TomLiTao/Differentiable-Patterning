import jax
import equinox as eqx
import time
import diffrax
from PDE.reaction_diffusion_advection.update import F

from Common.model.abstract_model import AbstractModel

class PDE_solver(AbstractModel):
	func: F	
	dt0: float
	def __init__(self,N_CHANNELS,PERIODIC,dx=0.1,dt=0.1,key=jax.random.PRNGKey(int(time.time()))):
		self.func = F(N_CHANNELS,PERIODIC,dx,key)
		self.dt0 = dt
	def __call__(self, ts, y0):
		solution = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
									   diffrax.Tsit5(),
									   t0=ts[0],t1=ts[-1],
									   dt0=self.dt0,
									   y0=y0,
									   max_steps=16**4,
									   stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
									   saveat=diffrax.SaveAt(ts=ts))
		return solution.ts,solution.ys
	
	def partition(self):
		func_diff,func_static = self.func.partition()
		where = lambda s:s.func
		total_diff,total_static = eqx.partition(self,eqx.is_array)
		total_diff = eqx.tree_at(where,total_diff,func_diff)
		total_static=eqx.tree_at(where,total_static,func_static)
		return total_diff,total_static
		

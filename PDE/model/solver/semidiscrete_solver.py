import jax
import equinox as eqx
import time
import diffrax
#from PDE.model.reaction_diffusion_advection.update import F

from Common.model.abstract_model import AbstractModel

class PDE_solver(AbstractModel):
	func: eqx.Module	
	dt0: float
	def __init__(self,F,dt=0.1):
		self.func = F
		self.dt0 = dt
	def __call__(self, ts, y0):
		solution = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
									   diffrax.Heun(),
									   t0=ts[0],t1=ts[-1],
									   dt0=self.dt0,
									   y0=y0,
									   max_steps=100*ts.shape[0],
									   #stepsize_controller=diffrax.ConstantStepSize(),
									   stepsize_controller=diffrax.PIDController(rtol=1e-1, atol=1e-2),
									   saveat=diffrax.SaveAt(ts=ts))
		return solution.ts,solution.ys
	
	def partition(self):
		func_diff,func_static = self.func.partition()
		where = lambda s:s.func
		total_diff,total_static = eqx.partition(self,eqx.is_array)
		total_diff = eqx.tree_at(where,total_diff,func_diff)
		total_static=eqx.tree_at(where,total_static,func_static)
		return total_diff,total_static
		

import jax
import jax.numpy as np
import equinox as eqx
from ABM.model.neural_slime import NeuralSlime
from ABM.model.agent_nn_angle import agent_nn as agent_nn_angle

class NeuralSlimeAngle(NeuralSlime):
	
	
	def __init__(self,*args,**kwargs):
		NeuralSlime.__init__(self,*args,**kwargs)
		self.Agent_nn = agent_nn_angle(self.N_CHANNELS)
	@eqx.filter_jit
	def update_velocities_and_pheremones(self,agents,pheremone_weights,pheremone_lattice):
		v_agent_nn = jax.vmap(self.Agent_nn,in_axes=0,out_axes=(1,1,1),axis_name="N_AGENTS") # vmap over agents 
		d_v_mag,d_pheremones,d_angle = v_agent_nn(pheremone_weights)
		
		v_angle = np.arctan2(agents[1][1],agents[1][0])
		v_mag = np.sqrt(agents[1][1]**2+agents[1][0]**2)
		#print(c_angle.shape)
		#print(d_angle.shape)
		v_angle = v_angle+self.dt*d_angle[:,0]
		v_mag = v_mag+self.dt*d_v_mag
		agent_vel = np.array([np.cos(v_angle),np.sin(v_angle)])*v_mag
		
		
		pos = np.rint(agents[0]).astype(int)
		smooth_func = jax.vmap(self.pheremone_diffuse,in_axes=0,out_axes=0,axis_name="pheremones")
		pheremone_lattice = pheremone_lattice.at[:,pos[0],pos[1]].set(jax.nn.relu(pheremone_lattice[:,pos[0],pos[1]]+self.dt*d_pheremones))
		pheremone_lattice = smooth_func(pheremone_lattice)
		pheremone_lattice = self.pheremone_decay(pheremone_lattice)
		return (agents[0],agent_vel),pheremone_lattice
	
	@eqx.filter_jit
	def __call__(self,agents,pheremone_lattice):
		pheremone_weights = self.sense_pheremones(agents, pheremone_lattice)
		agents,pheremone_lattice = self.update_velocities_and_pheremones(agents, pheremone_weights, pheremone_lattice)
		agents = self.update_positions(agents)
		return agents,pheremone_lattice
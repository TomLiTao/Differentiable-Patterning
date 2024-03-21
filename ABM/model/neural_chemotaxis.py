import jax
import jax.numpy as np
import equinox as eqx
from ABM.model.neural_slime import NeuralSlime
from ABM.model.agent_nn_angle import agent_nn


class NeuralChemotaxis(NeuralSlime):
	
	def __init__(self,*args,**kwargs):
		NeuralSlime.__init__(self,*args,**kwargs)
		self.Agent_nn = agent_nn(self.N_CHANNELS)
		
	@eqx.filter_jit
	def _sense_pheremones(self,agents,pheremone_lattice):
		"""
		Detect pheremones and gradients of pheremones at agent location
	
		Parameters
		----------
		agents : (array[2,n_agents],array[2,n_agents])
			Tuple containing positions and velocities of all agents.
		pheremone_lattice : array[channels,grid_size,grid_size]
			Lattice for storing concentration of signalling pheremones.
		sensor_angle : float, optional
			Angle between different sensor regions. The default is 0.6.
		sensor_length : float, optional
			Distance between agent and sensor regions. The default is 3.
	
		Returns
		-------
		pheremone_weights: (array[channels,n_agents],array[channels,n_agents],array[channels,n_agents])
			Sum of pheremone concentrations in each sensor region for each agent
	
		"""
		
		
		def sense_zone_ind(pos,vel):
			"""
			Sense pheremones and gradients of them at a given agent position and direction

			VMAP THIS OVER AGENTS
	
			Parameters
			----------
			pos : array[2]
				 positions
			vel : array[2]
				velocities
			
	
			Returns
			-------
			weights: array[3*channels + 2]
				array storing local pheremone averages, gradients of pheremones and orientation
			
	
			"""
			sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
			sobel_y = sobel_x.T
			c_angle = np.arctan2(vel[1],vel[0])
			cos_angle = np.cos(c_angle) #vel[0]/(np.sqrt(vel[0]**2+vel[1]**2+0.0000001))
			sin_angle = np.sin(c_angle) #vel[1]/(np.sqrt(vel[0]**2+vel[1]**2+0.0000001))
			p_x = cos_angle*sobel_x + sin_angle*sobel_y
			p_y = cos_angle*sobel_y - sin_angle*sobel_x
			c = np.rint(pos).astype(int)
			if self.PERIODIC:
				x_ind = np.stack((c[0]-1,c[0],c[0]+1))%self.GRID_SIZE
				y_ind = np.stack((c[1]-1,c[1],c[1]+1))%self.GRID_SIZE	
				_xs,_ys = np.meshgrid(x_ind,y_ind)
				id = np.sum(pheremone_lattice[:,_xs,_ys],axis=(1,2)) # [:] over channels
				w_x = np.sum(pheremone_lattice[:,_xs,_ys]*p_x,axis=(1,2)) # [:] over channels
				w_y = np.sum(pheremone_lattice[:,_xs,_ys]*p_y,axis=(1,2)) # [:] over channels
			else:
				padwidth=1#2*np.rint(self.sensor_length).astype(int)
				x_ind = np.stack((c[0]-1,c[0],c[0]+1)) + padwidth
				y_ind = np.stack((c[1]-1,c[1],c[1]+1)) + padwidth
				_xs,_ys = np.meshgrid(x_ind,y_ind)
				padded_pheremone_lattice = np.pad(pheremone_lattice,((0,0),(padwidth,padwidth),(padwidth,padwidth)),constant_values=0.0)
				id = np.sum(padded_pheremone_lattice[:,_xs,_ys],axis=(1,2)) # [:] over channels
				w_x = np.sum(pheremone_lattice[:,_xs,_ys]*p_x,axis=(1,2)) # [:] over channels
				w_y = np.sum(pheremone_lattice[:,_xs,_ys]*p_y,axis=(1,2)) # [:] over channels
			#weights = np.concatenate([id,w_x,w_y,cos_angle,sin_angle],axis=0)
			#return weights
			
			return id,w_x,w_y,cos_angle[np.newaxis],sin_angle[np.newaxis]
		v_sense = jax.vmap(sense_zone_ind,(0,0),(0,0,0,0,0),axis_name="N_AGENTS")
		#print(agents.shape)
		weights =v_sense(agents[0].T,agents[1].T)
		
		return weights
	
	@eqx.filter_jit
	def _update_velocities_and_pheremones(self,agents,pheremone_weights,pheremone_lattice):
		v_agent_nn = jax.vmap(self.Agent_nn,in_axes=0,out_axes=(1,1,1),axis_name="N_AGENTS") # vmap over agents 
		d_v_mag,d_pheremones,d_angle = v_agent_nn(pheremone_weights)
		pos = np.rint(agents[0]).astype(int)
		if self.PERIODIC:
			vel = agents[1]
		else:
			boundary_mask = np.logical_or(pos==0,pos==self.GRID_SIZE)

			vel = np.where(boundary_mask,-agents[1],agents[1])
		
		
		v_angle = np.arctan2(vel[1],vel[0])
		v_mag = np.sqrt(vel[1]**2+vel[0]**2)
		#print(c_angle.shape)
		#print(d_angle.shape)
		v_angle = v_angle+self.dt*d_angle[:,0]
		v_mag = np.clip(v_mag+self.dt*d_v_mag,a_min=-10*self.dt,a_max=10*self.dt)
		agent_vel = np.array([np.cos(v_angle),np.sin(v_angle)])*v_mag
		
		
		
		smooth_func = jax.vmap(self._pheremone_diffuse,in_axes=0,out_axes=0,axis_name="pheremones")
		pheremone_lattice = pheremone_lattice.at[:,pos[0],pos[1]].set(jax.nn.relu(pheremone_lattice[:,pos[0],pos[1]]+self.dt*d_pheremones))
		pheremone_lattice = smooth_func(pheremone_lattice)
		pheremone_lattice = self._pheremone_decay(pheremone_lattice)
		return (agents[0],agent_vel),pheremone_lattice
	
	# @eqx.filter_jit
	# def __call__(self,state):
	# 	print("State: ")
	# 	print(state)
	# 	agents,pheremone_lattice = state
	# 	print("Agents: ")
	# 	print(agents)
	# 	pheremone_weights = self._sense_pheremones(agents, pheremone_lattice)
	# 	agents,pheremone_lattice = self._update_velocities_and_pheremones(agents, pheremone_weights, pheremone_lattice)
	# 	agents = self._update_positions(agents)
	# 	state = (agents,pheremone_lattice)
	# 	return state

	@eqx.filter_jit
	def __call__(self,state):
		#print("State: ")
		#print(state)
		((agents_p,agents_v),pheremone_lattice) = state
		#print("Agent position: ")
		#print(agents_p)
		#print("Agent velocity: ")
		#print(agents_v)
		#print("Pheremone lattice: ")
		#print(pheremone_lattice)
		pheremone_weights = self._sense_pheremones((agents_p,agents_v), pheremone_lattice)
		((agents_p,agents_v),pheremone_lattice) = self._update_velocities_and_pheremones((agents_p,agents_v), pheremone_weights, pheremone_lattice)
		(agents_p,agents_v) = self._update_positions((agents_p,agents_v))
		state = ((agents_p,agents_v),pheremone_lattice)
		return state
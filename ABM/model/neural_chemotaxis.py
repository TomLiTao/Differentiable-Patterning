import jax
import jax.numpy as np
import equinox as eqx
from ABM.model.neural_slime import NeuralSlime
from ABM.model.agent_nn_vel import agent_nn

class NeuralChemotaxis(NeuralSlime):
	
	def __init__(self,*args,**kwargs):
		NeuralSlime.__init__(self,*args,**kwargs)
		self.Agent_nn = agent_nn(self.N_CHANNELS)
		
	@eqx.filter_jit
	def sense_pheremones(self,agents,pheremone_lattice):
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
			
	
			Parameters
			----------
			pos : array[n_agents,2]
				transposed positions
			vel : array[n_agents,2]
				transposed velocities
			angle_offset : float
				angle  for sensor region position. clockwise from velocity direction
	
			Returns
			-------
			pheremone_weight : array[n_agents]
				sum of pheremones at sensor for each agent
	
			"""
			sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
			sobel_y = sobel_x.T
			#c_angle = np.arctan2(vel[1],vel[0]) + angle_offset
			c = np.rint(pos).astype(int)
			if self.PERIODIC:
				x_ind = np.stack((c[0]-1,c[0],c[0]+1))%self.GRID_SIZE
				y_ind = np.stack((c[1]-1,c[1],c[1]+1))%self.GRID_SIZE	
				_xs,_ys = np.meshgrid(x_ind,y_ind)
				weights = np.sum(pheremone_lattice[:,_xs,_ys],axis=(1,2))
				w_x = np.sum(pheremone_lattice[:,_xs,_ys]*sobel_x,axis=(1,2))
				w_y = np.sum(pheremone_lattice[:,_xs,_ys]*sobel_y,axis=(1,2))
			else:
				padwidth=1#2*np.rint(self.sensor_length).astype(int)
				x_ind = np.stack((c[0]-1,c[0],c[0]+1)) + padwidth
				y_ind = np.stack((c[1]-1,c[1],c[1]+1)) + padwidth
				_xs,_ys = np.meshgrid(x_ind,y_ind)
				padded_pheremone_lattice = np.pad(pheremone_lattice,((0,0),(padwidth,padwidth),(padwidth,padwidth)),constant_values=0.0)
				weights = np.sum(padded_pheremone_lattice[:,_xs,_ys],axis=(1,2))
				w_x = np.sum(pheremone_lattice[:,_xs,_ys]*sobel_x,axis=(1,2))
				w_y = np.sum(pheremone_lattice[:,_xs,_ys]*sobel_y,axis=(1,2))
			return weights,w_x,w_y
	
		v_sense = jax.vmap(sense_zone_ind,(0,0),(0,0,0))
		weights,w_x,w_y =v_sense(agents[0].T,agents[1].T)
		return weights,w_x,w_y
	
	@eqx.filter_jit
	def __call__(self,agents,pheremone_lattice):
		pheremone_weights = self.sense_pheremones(agents, pheremone_lattice)
		agents,pheremone_lattice = self.update_velocities_and_pheremones(agents, pheremone_weights, pheremone_lattice)
		agents = self.update_positions(agents)
		return agents,pheremone_lattice
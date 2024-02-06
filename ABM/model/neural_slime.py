import jax
import jax.numpy as np
import equinox as eqx
from ABM.model.agent_nn_vel import agent_nn
from Common.model.abstract_model import AbstractModel # Inherit save, load, partition and combine
import time


class NeuralSlime(AbstractModel):
	Agent_nn: agent_nn
	N_AGENTS: int
	GRID_SIZE: int
	N_CHANNELS: int 
	dt: float
	sensor_angle: float
	sensor_length: float
	gaussian_blur: int
	decay_rate: float
	PERIODIC: bool


	def __init__(self,
			     N_AGENTS,
				 GRID_SIZE,
				 N_CHANNELS,
				 dt=0.1,
				 sensor_angle=0.6,
				 sensor_length=3,
				 gaussian_blur=0,
				 decay_rate=0.99,
				 PERIODIC=True,
				 key=jax.random.PRNGKey(int(time.time()))):
		"""
		

		Parameters
		----------
		N_AGENTS : int
			Number of interacting agents.
		GRID_SIZE : int
			size of the square grid that the agents move around in.
		N_CHANNELS : int
			Number of distinct pheremone species.
		dt : TYPE, optional
			timestep size for incrementing position and pheremone concentrations. The default is 0.1.
		sensor_angle : float, optional
			angle seperating different pheremone sensors in front of agent. The default is 0.6.
		sensor_length : float, optional
			how far in front of agent are the pheremone sensors. The default is 3.
		gaussian_blur : int, optional
			Size of kernel for diffusing pheremone signals. If 0, no diffusion. The default is 3.
		decay_rate : float, optional
			Decay rate for pheremone signal. The default is 0.99.
		PERIODIC: boolean, optional
			Flag for whether to have periodic boundary conditions or not. The default is True.
		key : TYPE, optional
			Jax random number key. The default is jax.random.PRNGKey(int(time.time())).

		Returns
		-------
		None.

		"""
		#self.Agent_nn = jax.vmap(agent_nn(N_CHANNELS,key),in_axes=0,out_axes=(0,0),axis_name="N_AGENTS") # vmap over agents 
		self.Agent_nn = agent_nn(N_CHANNELS,key) 
		self.N_AGENTS = N_AGENTS
		self.GRID_SIZE = GRID_SIZE
		self.N_CHANNELS = N_CHANNELS
		self.dt = dt
		self.sensor_angle = sensor_angle
		self.sensor_length= sensor_length
		self.gaussian_blur=gaussian_blur
		self.decay_rate = decay_rate
		self.PERIODIC = PERIODIC
	def init_state(self,key=jax.random.PRNGKey(int(time.time())),zero_pheremone=True):
		_w=2
		#print(key)
		key1,key2,key3,key4 = jax.random.split(key,4)
		agent_pos = jax.random.uniform(key1,shape=[2,self.N_AGENTS],minval=_w,maxval=self.GRID_SIZE-_w)
		agent_vel = jax.random.uniform(key2,shape=[2,self.N_AGENTS],minval=-1,maxval=1)
		if zero_pheremone:
			pheremone_lattice = np.zeros((self.N_CHANNELS,self.GRID_SIZE,self.GRID_SIZE))
		else:
			pheremone_lattice = jax.random.uniform(key3,shape=[self.N_CHANNELS,self.GRID_SIZE,self.GRID_SIZE],minval=0,maxval=1)
		state = ((agent_pos,agent_vel),pheremone_lattice) 
		return state
	@eqx.filter_jit
	def _update_positions(self,agents):
		"""
		Updates positions of all agents. Implements periodic boundary condition

		Parameters
		----------
		agents : (array[2,n_agents],array[2,n_agents])
			Tuple containing positions and velocities of all agents.
		Returns
		-------
		agents : (array[2,n_agents],array[2,n_agents])
			Tuple containing positions and velocities of all agents.

		"""
		if self.PERIODIC:
			return (agents[0]+agents[1]*self.dt)%self.GRID_SIZE,agents[1]
		else:
			return np.clip(agents[0]+agents[1]*self.dt,a_min=0,a_max=self.GRID_SIZE),agents[1]
	@eqx.filter_jit
	def _sense_pheremones(self,agents,pheremone_lattice):
		"""
		Detect pheremones in front of each agent
	
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
		
		
		def sense_zone_ind(pos,vel,angle_offset):
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
			c_angle = np.arctan2(vel[1],vel[0]) + angle_offset
			c = np.rint(pos + self.sensor_length*np.array([np.cos(c_angle),np.sin(c_angle)])).astype(int)
			if self.PERIODIC:
				x_ind = np.stack((c[0]-1,c[0],c[0]+1))%self.GRID_SIZE
				y_ind = np.stack((c[1]-1,c[1],c[1]+1))%self.GRID_SIZE	
				_xs,_ys = np.meshgrid(x_ind,y_ind)
				weights = np.sum(pheremone_lattice[:,_xs,_ys],axis=(1,2))
			else:
				padwidth=6#2*np.rint(self.sensor_length).astype(int)
				x_ind = np.stack((c[0]-1,c[0],c[0]+1)) + padwidth
				y_ind = np.stack((c[1]-1,c[1],c[1]+1)) + padwidth
				_xs,_ys = np.meshgrid(x_ind,y_ind)
				padded_pheremone_lattice = np.pad(pheremone_lattice,((0,0),(padwidth,padwidth),(padwidth,padwidth)),constant_values=0.0)
				weights = np.sum(padded_pheremone_lattice[:,_xs,_ys],axis=(1,2))
			return weights
	
		sense_zone_vec = jax.vmap(sense_zone_ind,(0,0,None),(0))
		return sense_zone_vec(agents[0].T,agents[1].T,0),sense_zone_vec(agents[0].T,agents[1].T,self.sensor_angle),sense_zone_vec(agents[0].T,agents[1].T,-self.sensor_angle)
	@eqx.filter_jit	
	def _pheremone_diffuse(self,pheremone_lattice):
		"""
		NOTE: VMAP THIS OVER PHEREMONE CHANNELS		

		Parameters
		----------
		pheremone_lattice : array[GRID_SIZE,GRID_SIZE]
			2d array to smooth

		Returns
		-------
		pheremone_lattice : array[GRID_SIZE,GRID_SIZE]
			2d smoothed array

		"""
		if self.gaussian_blur==0:
			return pheremone_lattice
		else:
			x = np.linspace(-self.gaussian_blur, self.gaussian_blur, 2*self.gaussian_blur+1)
			window = jax.scipy.stats.norm.pdf(x) * jax.scipy.stats.norm.pdf(x[:, None])
			am = 0.1

			return am*jax.scipy.signal.convolve2d(pheremone_lattice, window, mode='same',boundary="fill") + (1-am)*pheremone_lattice
	
	@eqx.filter_jit
	def _pheremone_decay(self,pheremone_lattice):
		return pheremone_lattice*self.decay_rate
	
	@eqx.filter_jit
	def _update_velocities_and_pheremones(self,agents,pheremone_weights,pheremone_lattice):
		v_agent_nn = jax.vmap(self.Agent_nn,in_axes=0,out_axes=(0,0),axis_name="N_AGENTS") # vmap over agents 
		agent_vel,d_pheremones = v_agent_nn(pheremone_weights)

		agent_vel = agent_vel.T
		d_pheremones = d_pheremones.T
		pos = np.rint(agents[0]).astype(int)
		smooth_func = jax.vmap(self._pheremone_diffuse,in_axes=0,out_axes=0,axis_name="pheremones")
		
		pheremone_lattice = pheremone_lattice.at[:,pos[0],pos[1]].set(jax.nn.relu(pheremone_lattice[:,pos[0],pos[1]]+self.dt*d_pheremones))
		pheremone_lattice = smooth_func(pheremone_lattice)
		pheremone_lattice = self._pheremone_decay(pheremone_lattice)
		return (agents[0],agent_vel),pheremone_lattice
	@eqx.filter_jit
	def __call__(self,state):
		agents,pheremone_lattice = state
		pheremone_weights = self._sense_pheremones(agents, pheremone_lattice)
		agents,pheremone_lattice = self._update_velocities_and_pheremones(agents, pheremone_weights, pheremone_lattice)
		agents = self._update_positions(agents)
		state = (agents,pheremone_lattice)
		return state
	

	# @eqx.filter_jit
	# def __call__(self,state):
	# 	((agents_p,agents_v),pheremone_lattice) = state
	# 	pheremone_weights = self._sense_pheremones((agents_p,agents_v), pheremone_lattice)
	# 	(agents_p,agents_v),pheremone_lattice = self._update_velocities_and_pheremones((agents_p,agents_v), pheremone_weights, pheremone_lattice)
	# 	(agents_p,agents_v) = self._update_positions((agents_p,agents_v))
	# 	state = ((agents_p,agents_v),pheremone_lattice)
	# 	return state



import jax
import jax.numpy as np
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from functools import partial
import time

def setup(n_agents,grid_size,key):
	"""
	
	Parameters
	----------
	n_agents : int
		Number of agents to initialise.
	grid_size : int
		Size of lattice.
	key : PRNG key
		RNG key.

	Returns
	-------
	(agent_pos,agent_vel) : (array[2,n_agents],array[2,n_agents])
		Tuple containing positions and velocities of all agents.
	pheremone_lattice : array[grid_size,grid_size]
		Lattice for storing concentration of signalling pheremones.

	"""
	
	_w=2
	v=1
	v_angle = jax.random.uniform(key,shape=[n_agents],minval=-np.pi,maxval=np.pi)
	#plt.hist(v_angle)
	#plt.show()
	agent_pos = jax.random.uniform(key,shape=[2,n_agents],minval=_w,maxval=grid_size-_w)
	agent_vel = v*np.asarray([np.cos(v_angle),np.sin(v_angle)])
	#plt.scatter(agent_vel[0],agent_vel[1])
	#plt.show()
	pheremone_lattice = np.zeros((grid_size,grid_size))
	return (agent_pos,agent_vel),pheremone_lattice
@jax.jit
def update_positions(agents,grid_size):
	"""
	Updates positions of all agents. Implements periodic boundary condition

	Parameters
	----------
	agents : (array[2,n_agents],array[2,n_agents])
		Tuple containing positions and velocities of all agents.
	grid_size : int
		Size of lattice.
	Returns
	-------
	agents : (array[2,n_agents],array[2,n_agents])
		Tuple containing positions and velocities of all agents.

	"""
	return (agents[0]+agents[1])%grid_size,agents[1]


@jax.jit
def sense_pheremones(agents,pheremone_lattice,sensor_angle=0.6,sensor_length=3):
	"""
	Detect pheremones in front of each agent

	Parameters
	----------
	agents : (array[2,n_agents],array[2,n_agents])
		Tuple containing positions and velocities of all agents.
	pheremone_lattice : array[grid_size,grid_size]
		Lattice for storing concentration of signalling pheremones.
	sensor_angle : float, optional
		Angle between different sensor regions. The default is 0.6.
	sensor_length : float, optional
		Distance between agent and sensor regions. The default is 3.

	Returns
	-------
	pheremone_weights: (array[n_agents],array[n_agents],array[n_agents])
		Sum of pheremone concentrations in each sensor region for each agent

	"""
	grid_size = pheremone_lattice.shape[0]
	
	
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
		c = np.rint(pos + sensor_length*np.array([np.cos(c_angle),np.sin(c_angle)])).astype(int)
		x_ind = np.stack((c[0]-1,c[0],c[0]+1))%grid_size
		y_ind = np.stack((c[1]-1,c[1],c[1]+1))%grid_size	
		_xs,_ys = np.meshgrid(x_ind,y_ind)
		return np.sum(pheremone_lattice[_xs,_ys])

	sense_zone_vec = jax.vmap(sense_zone_ind,(0,0,None),(0))
	return sense_zone_vec(agents[0].T,agents[1].T,0),sense_zone_vec(agents[0].T,agents[1].T,sensor_angle),sense_zone_vec(agents[0].T,agents[1].T,-sensor_angle)
	



@jax.jit
def update_velocities(agents,pheremone_weights,key,zero_thresh=0.1,v_abs=0.1,d_angle=0.5):
	"""
	Probabilistically updates each agent velocity based on pheremone weights

	Parameters
	----------
	agents : (array[2,n_agents],array[2,n_agents])
		Tuple containing positions and velocities of all agents.
	pheremone_weights: (array[n_agents],array[n_agents],array[n_agents])
		Sum of pheremone concentrations in each sensor region for each agent
	key : PRNG key
		RNG key.
	zero_thresh : float, optional
		How close to 0 pheremone weight needs to be for agent to ignore it completely. The default is 0.1.
	v_abs : float, optional
		Speed of agents. The default is 0.1.
	d_angle : float, optional
		Speed of angle change. The default is 0.5.

	Returns
	-------
	agents : (array[2,n_agents],array[2,n_agents])
		Tuple containing positions and velocities of all agents.

	"""
	A = agents[0].shape[1]
	u = jax.random.uniform(key,shape=[A],minval=0,maxval=1)
	W_c,W_l,W_r = pheremone_weights
	W_total = W_c+W_l+W_r
	W_c/=W_total
	W_l/=W_total
	W_r/=W_total
	
	W_c = np.where(W_total<zero_thresh,0.98,W_c)
	W_l = np.where(W_total<zero_thresh,0.01,W_c)
	W_r = np.where(W_total<zero_thresh,0.01,W_c)
	
	mask_c = (u<W_c).astype(int)
	mask_l = ((u>=W_c) & (u<W_c+W_l)).astype(int)
	mask_r = ((u>=W_c+W_l)).astype(int)
	dw = mask_l-mask_r
	v_angle = np.arctan2(agents[1][1],agents[1][0])

	return agents[0],v_abs*np.array([np.cos(v_angle+dw*d_angle),np.sin(v_angle+dw*d_angle)])

	


@jax.jit
def update_pheremone(agents,pheremone_lattice,decay_rate=0.99):
	pos = np.rint(agents[0]).astype(int)
	x = np.linspace(-3, 3, 7)
	window = jax.scipy.stats.norm.pdf(x) * jax.scipy.stats.norm.pdf(x[:, None])
	#pheremone_lattice = jax.scipy.signal.convolve2d(pheremone_lattice, window, mode='same',boundary="wrap")
	pheremone_lattice = pheremone_lattice.at[pos[0],pos[1]].set(pheremone_lattice[pos[0],pos[1]]+1)
	return pheremone_lattice*decay_rate




def my_animate(img):
  """
    Boilerplate code to produce matplotlib animation

    Parameters
    ----------
    img : float32 or int array [t,x,y,rgb]
      img must be float in range [0,1] or int in range [0,255]

  """
  #img = np.clip(img,0,1)
  frames = [] # for storing the generated images
  fig = plt.figure()
  t_max = np.max(img)
  t_min = np.min(img)
  for i in range(img.shape[0]):
    frames.append([plt.imshow(img[i],vmin=t_min,vmax=t_max,animated=True)])

  ani = animation.ArtistAnimation(fig, frames, interval=1, blit=True,
                                repeat_delay=0)
  
  plt.show()









def single_step():
	grid_size = 100
	agents,pheremone_lattice = setup(1,grid_size)
	key = jax.random.PRNGKey(0)
	t1 = time.time()
	agents = update_positions(agents, grid_size)
	t2 = time.time()
	pheremone_lattice = update_pheremone(agents, pheremone_lattice)
	t3 = time.time()
	pheremone_weights = sense_pheremones(agents, pheremone_lattice)
	t4 = time.time()
	agents = update_velocities(agents, pheremone_weights,key)
	t5 = time.time()
	print("Updating positions: "+str(t2-t1))
	print("Updating Pheremones: "+str(t3-t2))
	print("Sensing Pheremones: "+str(t4-t3))
	print("Updating velocities: "+str(t5-t4))
	print(pheremone_lattice)

#single_step()

def run():
	key = jax.random.PRNGKey(12)
	grid_size = 128
	agent_number = 200
	its = 1000
	agents,pheremone_lattice = setup(agent_number,grid_size,key)
	
	
	trajectory = numpy.zeros((its,grid_size,grid_size))
	for i in tqdm(range(its)):
		key = jax.random.PRNGKey(i) 	
		pheremone_lattice = update_pheremone(agents, pheremone_lattice)	
		pheremone_weights = sense_pheremones(agents, pheremone_lattice) 	
		agents = update_velocities(agents, pheremone_weights,key)
		agents = update_positions(agents, grid_size)	
		#pheremone_lattice = 0.98*pheremone_lattice
		trajectory[i] = pheremone_lattice
		
	my_animate(trajectory)
	
run()
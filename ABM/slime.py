import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
#simple agent/particle based slime mold simulation


class SlimeAgent(object):
	def __init__(self,S=128,v=0.3):
		_w = 2
		self.pos = np.random.uniform(_w,S-_w,2)
		v_angle = np.random.uniform(-np.pi,np.pi,1)
		#print(v_angle)
		self.vel = v*np.array([np.cos(v_angle)[0],np.sin(v_angle)[0]])
		self.S = S
		self.d_angle = 0.3 # angle changing velocity
		self.vel_abs = v
	def update_pos(self):
		
		self.pos = np.mod(self.pos + self.vel,self.S-1)
	
	def get_pos_int(self):
		return np.rint(self.pos).astype(int)

	def sense_pheremone(self,pheremone_lattice,sensor_angle=0.6,sensor_length=3,width=3,zero_thresh=0.1):
		
	
		def sense_zone(angle_offset,sensor_length,width):
			c_angle = np.arctan2(self.vel[1],self.vel[0]) + angle_offset
			c = np.rint(self.pos + sensor_length*np.array([np.cos(c_angle),np.sin(c_angle)])).astype(int)
			s=0
			for x in range(c[0]-width,c[0]+width):
				for y in range(c[1]-width,c[1]+width):
					s+=pheremone_lattice[x%self.S,y%self.S]
			return s
		#sense_zone_jit = jax.jit(sense_zone)
		w_c = sense_zone(0,sensor_length,width)
		w_l = sense_zone(sensor_angle,sensor_length,width)
		w_r = sense_zone(-sensor_angle,sensor_length,width)
		w_total = w_c+w_r+w_l
		if w_total<zero_thresh:
			w_c=0.01#1/3
			w_r=0.98#1/3
			w_l=0.01#1/3
		else:
			w_c/=w_total
			w_r/=w_total
			w_l/=w_total
# 		if (w_c > w_l) and (w_c > w_r):
# 			#more pheremone at front
# 			dw = 0
# 		elif w_r > w_l:
# 			dw = -1
# 		elif w_l > w_r:
# 			dw = 1
# 		else:
# 			dw=0
# 		
		dw = np.random.choice([-1,0,1],p=[w_r,w_c,w_l])
		v_angle = np.arctan2(self.vel[1],self.vel[0])
		
		self.vel = self.vel_abs*np.array([np.cos(v_angle+dw*self.d_angle),np.sin(v_angle+dw*self.d_angle)])
		return v_angle
class SlimeMold(object):
	def __init__(self,agents,S=128):
		self.agents = [SlimeAgent(S) for i in range(agents)]
		self.S = S
		self.pheremone_lattice = np.zeros((S,S))
		 
	def update_pos(self):
		for agent in self.agents:
			agent.update_pos()
	

			



	def run(self,steps):
		smooth_kernel = np.array([[1,1,1],
								  [1,20,1],
								  [1,1,1]])/28.0
		trajectory = np.zeros((steps,self.S,self.S))
		pheremone_trajectory = np.zeros((steps,self.S,self.S))
		v_angles = []
		for i in tqdm(range(steps)):			
			
			for agent in self.agents:
				pos = agent.get_pos_int()
				self.pheremone_lattice[pos[0],pos[1]]+=1
				v_angles.append(agent.sense_pheremone(self.pheremone_lattice))
				trajectory[i,pos[0],pos[1]]+=1
			self.pheremone_lattice*=0.99
			#self.pheremone_lattice = sp.signal.convolve2d(self.pheremone_lattice,smooth_kernel,mode="same",boundary="wrap")
			self.update_pos()
			pheremone_trajectory[i]=self.pheremone_lattice
		plt.hist(v_angles,bins=100)
		plt.show()
		return trajectory,pheremone_trajectory



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

  ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True,
                                repeat_delay=0)
  
  plt.show()



slime = SlimeMold(300,128)
trajectory,pheremones = slime.run(4000)
my_animate(trajectory)
my_animate(pheremones)
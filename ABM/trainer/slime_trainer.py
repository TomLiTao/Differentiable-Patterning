import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import datetime
import time
from PDE.trainer.data_augmenter_pde import DataAugmenterPDE # Fine to use this for now
import NCA.trainer.loss as loss
#from ABM.model.neural_slime import NeuralSlime
#from NCA_JAX.model.boundary import NCA_boundary
#from PDE.trainer.tensorboard_log import PDE_Train_log
#from PDE.trainer.optimiser import non_negative_diffusion
#from PDE.solver.semidiscrete_solver import PDE_solver
#import diffrax
from tqdm import tqdm

class SlimeTrainer(object):
	"""
	Train random initial configurations of slime mold agents to produce an image
	"""
	def __init__(self,
			     nslime,
				 data,
				 model_filename,
				 DATA_AUGMENTER = DataAugmenterPDE,
				 directory = "models/"):
		self.nslime = nslime
		#self.OBS_CHANNELS = self.nslime.N_CHANNELS
		self.OBS_CHANNELS = data[0].shape[1]
		print("Observable Channels: "+str(self.OBS_CHANNELS))
		# Set up data and data augmenter class
		self.DATA_AUGMENTER = DATA_AUGMENTER(data)
		self.DATA_AUGMENTER.data_init()
		self.BATCHES = len(data)
		print("Batches = "+str(self.BATCHES))
		if model_filename is None:
			self.model_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			self.IS_LOGGING = False
		else:
			self.model_filename = model_filename
			self.IS_LOGGING = True
			self.LOG_DIR = "logs/"+self.model_filename+"/train"
			#self.LOGGER = NCA_Train_log(self.LOG_DIR, data)
			print("Logging training to: "+self.LOG_DIR)
		self.directory = directory
		self.MODEL_PATH = directory+self.model_filename
		print("Saving model to: "+self.MODEL_PATH)
		
	def loss_func(self,x,y,key):
		"""
		NOTE: VMAP THIS OVER BATCHES TO HANDLE DIFFERENT SIZES OF GRID IN EACH BATCH
	
		Parameters
		----------
		x : float32 array [N,CHANNELS,_,_]
			pheremone state
		y : float32 array [N,OBS_CHANNELS,_,_]
			data
		Returns
		-------
		loss : float32 array [N]
			loss for each timestep of trajectory
		"""
		#x_obs = x[:self.OBS_CHANNELS]
		#y_obs = y[:self.OBS_CHANNELS]
		#x_obs = x_obs[jnp.newaxis]
		#y_obs = y_obs[jnp.newaxis]
		#return loss.vgg(x_obs,y_obs,key)
		x_obs = x[:self.OBS_CHANNELS]
		y_obs = y[:self.OBS_CHANNELS]
		return loss.euclidean(x_obs,y_obs)
	
	def train(self,
		      t,
			  iters,
			  optimiser=None,
			  WARMUP=64,
			  key=jax.random.PRNGKey(int(time.time()))):
		
		
		def make_step(nslime,target,t,opt_state,key):
			
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(nslime_diff,nslime_static,target,t,key):
				nslime = eqx.combine(nslime_diff,nslime_static)
				#v_nslime = jax.vmap(nslime,in_axes=(0,0),out_axes=0,axis_name="N")
				def nslime_step_wrapper(carry,j):# function of type a,b->a
					agents,pheremone_lattice=carry
					agents,pheremone_lattice = nslime(agents,pheremone_lattice)
					return (agents,pheremone_lattice),None
				agents,pheremone_lattice=nslime.init_state()
				
				(agents,pheremone_lattice),_=jax.lax.scan(nslime_step_wrapper,(agents,pheremone_lattice),xs=jnp.arange(t//2))
				loss_1 = self.loss_func(pheremone_lattice, target,key)
				(agents,pheremone_lattice),_=jax.lax.scan(nslime_step_wrapper,(agents,pheremone_lattice),xs=jnp.arange(t//2))
				loss_2 = self.loss_func(pheremone_lattice, target,key)
				mean_loss = jnp.mean(loss_1+loss_2)
				return mean_loss,(pheremone_lattice,loss_1+loss_2)
			
			nslime_diff,nslime_static = nslime.partition()
			loss_x,grads = compute_loss(nslime_diff,nslime_static,target,t,key)
			updates,opt_state = self.OPTIMISER.update(grads, opt_state, nslime_diff)
			nslime = eqx.apply_updates(nslime,updates)
			return nslime,opt_state,loss_x
		
		# Initialise Training
		nslime = self.nslime
		agents,pheremone_lattice=nslime.init_state()
		print("Pheremone Lattice: ")
		print(pheremone_lattice.shape)
		nslime_diff,nslime_static = nslime.partition()
		if optimiser is None:
			schedule = optax.exponential_decay(5e-2, transition_steps=iters, decay_rate=0.99)
			self.OPTIMISER = optax.adamw(schedule)
			#self.OPTIMISER = non_negative_diffusion(learn_rate=1e-2,iters=iters)
		else:
			self.OPTIMISER = optimiser
		opt_state = self.OPTIMISER.init(nslime_diff)
		
		best_loss = 100000000
		loss_thresh = 1e16
		model_saved = False
		error = 0
		error_at = 0
		target = self.DATA_AUGMENTER.return_true_data()[0][0]
		print("Data format: ")
		print(target.shape)
		for i in tqdm(range(iters)):
			key = jax.random.fold_in(key,i)
			nslime,opt_state,(mean_loss,(x,loss))= make_step(nslime,target,t,opt_state,key)
			
			
			
			
			
			
			if jnp.isnan(mean_loss):
				error = 1
				error_at=i
				break
			elif any(list(map(lambda x: jnp.any(jnp.isnan(x)), x))):
				error = 2
				error_at=i
				break
			elif mean_loss>loss_thresh:
				error = 3
				error_at=i
				break
			
			if error==0:
				if i>WARMUP:
					if mean_loss<best_loss:
						model_saved=True
						self.nslime = nslime
						self.nslime.save(self.MODEL_PATH,overwrite=True)
						best_loss = mean_loss
						tqdm.write("--- Model saved at "+str(i)+" epochs with loss "+str(mean_loss)+" ---")
			
		if error==0:
			print("Training completed successfully")
		elif error==1:
			print("|-|-|-|-|-|-  Loss reached NaN at step "+str(error_at)+" -|-|-|-|-|-|")
		elif error==2:
			print("|-|-|-|-|-|-  X reached NaN at step "+str(error_at)+" -|-|-|-|-|-|")
		elif error==3:
			print( "|-|-|-|-|-|-  Loss exceded "+str(loss_thresh)+" at step "+str(error_at)+", optimisation probably diverging  -|-|-|-|-|-|")
			
			
			
			
			
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU") # Force tensorflow not to use GPU, as it's only logging data
import numpy as np
from PDE.model.reaction_diffusion_advection.visualize import plot_weight_kernel_boxplot,plot_weight_matrices
from Common.trainer.abstract_tensorboard_log import Train_log
from einops import rearrange



class PDE_Train_log(Train_log):
	"""
		Class for logging training behaviour of PDE_Trainer classes, using tensorboard
	"""

	def log_model_parameters(self, model, i):
		with self.train_summary_writer.as_default():
			weight_matrix_figs = plot_weight_matrices(model)
			tf.summary.image("Weight matrices",np.array(weight_matrix_figs)[:,0],step=i,max_outputs=10)
			
			kernel_weight_figs = plot_weight_kernel_boxplot(model)
			tf.summary.image("Input weights per channel",np.array(kernel_weight_figs)[:,0],step=i,max_outputs=10)
				

	def log_model_outputs(self, x, i):
		#pass
		#print(x.shape)
		#print(len(x))
		#print(x[0].shape)
		N_SAMPLES = 4

		with self.train_summary_writer.as_default():
			BATCHES = len(x)
			for b in range(BATCHES):
#				if self.RGB_mode=="RGB":
		# 			#tf.summary.image('Trajectory batch '+str(b),np.einsum("ncxy->nxyc",x[b,:,:3,...]),step=i,max_outputs=x.shape[0])
				tf.summary.image('Trajectory batch '+str(b),rearrange(x[b][::x[b].shape[0]//N_SAMPLES,:3,...],"n c x y -> n x y c"),step=i,max_outputs=N_SAMPLES)
				if x[0].shape[1] > 4:
					hidden_channels = x[b][::x[b].shape[0]//N_SAMPLES,3:]
					extra_zeros = (-hidden_channels.shape[1])%3
					hidden_channels = np.pad(hidden_channels,((0,0),(0,extra_zeros),(0,0),(0,0)))
		
					tf.summary.image('Trajectory batch '+str(b)+', hidden channels',rearrange(hidden_channels, "N (Z C) X Y -> N (Z X) Y C",C=3),step=i,max_outputs=N_SAMPLES)


	
	def tb_training_end_log(self,pde,x,t,boundary_callback,write_images=True):
		"""
		

			Log trained NCA model trajectory after training

		"""

		#print(nca)
		with self.train_summary_writer.as_default():
			trs = []
			trs_h = []
			
			for b in range(len(x)):
				
				_,Y =pde(np.linspace(0,t,t+1),x[b][0])
				Y_h = []
				
				for i in range(t):
					y_h = Y[i][4:]
					extra_zeros = (-y_h.shape[0])%3
					y_h = np.pad(y_h,((0,extra_zeros),(0,0),(0,0)))
					y_h = np.reshape(y_h,(3,-1,y_h.shape[-1]))
					Y_h.append(y_h)
					#print(t_h.shape)
				trs.append(Y)
				trs_h.append(Y_h)
			for i in range(t):
				for b in range(len(x)):
					
					tf.summary.image("Final PDE trajectory, batch "+str(b),np.einsum("ncxy->nxyc",trs[b][i][np.newaxis,:3,...]),step=i)
					tf.summary.image("Final PDE trajectory hidden channels, batch "+str(b),np.einsum("ncxy->nxyc",trs_h[b][i][np.newaxis,...]),step=i)
					
				
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU") # Force tensorflow not to use GPU, as it's only logging data
import einops
import numpy as np
from ABM.ABM_visualiser import *
from Common.trainer.abstract_tensorboard_log import Train_log

class ABM_Train_log(Train_log):
    def __init__(self,log_dir,data,RGB_mode="RGB"):
        """
			Initialises the tensorboard logging of training.
			Writes some initial information. Very similar to setup_tb_log_single, but designed for sequence modelling

		"""

        self.LOG_DIR = log_dir
        self.RGB_mode = RGB_mode


        train_summary_writer = tf.summary.create_file_writer(self.LOG_DIR)

        #--- Log the target image and initial condtions
        with train_summary_writer.as_default():
            #for b in range(data[1].shape[0]):
                #tf.summary.image("True sequence pheremone",np.einsum("ncxy->nxyc",data[1][b,:,:3,::-1,...]),step=b)
            for i in range(data[1].shape[1]):
                tf.summary.image("True sequence position",parallel_scatter_wrap(data[0][0][:,i]),step=i)
                tf.summary.image("True sequence pheremone",np.einsum("ncxy->nxyc",data[1][:,i,:3,::-1,...]),step=i)
                    
            
        self.train_summary_writer = train_summary_writer
    def log_model_parameters(self, model, i):
        
        w_0 = model.Agent_nn.layers[1].weight
        b_0 = model.Agent_nn.layers[1].bias

        w_ph = model.Agent_nn.layers_pheremone[0].weight
        b_ph = model.Agent_nn.layers_pheremone[0].bias

        w_vel_mag = model.Agent_nn.layers_velocity_mag[0].weight

        w_angle = model.Agent_nn.layers_d_angle[0].weight


        with self.train_summary_writer.as_default():
            tf.summary.histogram('Input layer weights',w_0,step=i)
            tf.summary.histogram('Input layer bias',b_0,step=i)
            tf.summary.histogram('Pheremone layer weights',w_ph,step=i)
            tf.summary.histogram('Pheremone layer bias',b_ph,step=i)

            tf.summary.histogram('Velocity layer weights',w_vel_mag,step=i)	
            tf.summary.histogram('Angle layer weights',w_angle,step=i)	
            weight_matrix_figs = plot_weight_matrices(model)
            tf.summary.image("Weight matrices",np.array(weight_matrix_figs)[:,0],step=i)
    
    def log_model_outputs(self, X, i):
        #return super().log_model_outputs(x, i)
        with self.train_summary_writer.as_default():
            BATCHES = X[1].shape[0]
            for b in range(BATCHES):
                tf.summary.image("Trajectory batch "+str(b),parallel_scatter_wrap(X[0][0][b]),step=i,max_outputs=X[0][0].shape[1])
                ph = np.pad(X[1][b],((0,0),(0,(-X[1][b].shape[1])%3),(0,0),(0,0)))
                #tf.summary.image("Pheremone state, batch "+str(b),np.einsum("ncxy->nyxc",X[1][b,:,:3,...]),step=i,max_outputs=X[1].shape[1])
                tf.summary.image("Pheremone state, batch "+str(b),
                                 einops.rearrange(ph,"n (cw c) x y -> n (cw x) y  c",c=3)[:,:,::-1]
                                 ,step=i,max_outputs=X[1].shape[1])
        #return None
    

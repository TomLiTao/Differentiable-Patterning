import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU") # Force tensorflow not to use GPU, as it's only logging data
import numpy as np
import jax.tree_util as jtu
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
            tf.summary.image("Weight matrices",np.array(weight_matrix_figs)[:,0],step=i,max_outputs=len(weight_matrix_figs))
            
            kernel_weight_figs = plot_weight_kernel_boxplot(model)
            tf.summary.image("Input weights per channel",np.array(kernel_weight_figs)[:,0],step=i,max_outputs=len(kernel_weight_figs))
                

    def log_model_outputs(self, x, i):

        N_SAMPLES = 4
        #x = 0.5*(x+1)
        x = jtu.tree_map(lambda x: 0.5*(x+1),x)
        
        with self.train_summary_writer.as_default():
            BATCHES = len(x)
            tf.summary.image('Training outputs ',rearrange(x,"b n c x y ->b n x y c")[:,-1,:,:,:3],step=i,max_outputs=BATCHES)
            
            if x[0].shape[1] > 4:
                hidden_channels = []
                for b in range(BATCHES):
                    h = x[b][-1,3:]
                    extra_zeros = (-h.shape[0])%3
                    hidden_channels.append(np.pad(h,((0,extra_zeros),(0,0),(0,0))))
                tf.summary.image('Training outputs hidden channels',rearrange(hidden_channels, "B (Z C) X Y ->B (Z X) Y C",C=3),step=i,max_outputs=BATCHES)
#				if self.RGB_mode=="RGB":
        # 			#tf.summary.image('Trajectory batch '+str(b),np.einsum("ncxy->nxyc",x[b,:,:3,...]),step=i,max_outputs=x.shape[0])
                #tf.summary.image('Trajectory '+str(b),rearrange(x,"b n c x y ->b n x y c")[:,-1,:,:,:3],step=i,max_outputs=1)
        
                    #tf.summary.image('Trajectory batch '+str(b)+', hidden channels',rearrange(hidden_channels, "N (Z C) X Y -> N (Z X) Y C",C=3),step=i,max_outputs=1)


    
    def tb_training_end_log(self,pde,x,t,boundary_callback,write_images=True):
        """
        

            Log trained NCA model trajectory after training

        """

        #print(nca)
        with self.train_summary_writer.as_default():
            trs = []
            trs_h = []
            
            for b in range(len(x)):
                
                _,Y =pde(np.linspace(0,t,t+1),x[b])
                Y = 0.5*(Y+1)
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
            trs = np.array(trs)
            trs_h = np.array(trs_h)
            for i in range(t):
                tf.summary.image("Final PDE trajectory",rearrange(trs,"B N C X Y->B N X Y C")[:,i,:,:,:3],step=i)
                tf.summary.image("Final PDE trajectory hidden channels",rearrange(trs_h,"B N C X Y->B N X Y C")[:,i],step=i)
                #for b in range(len(x)):
                    
                #    tf.summary.image("Final PDE trajectory, batch "+str(b),np.einsum("ncxy->nxyc",trs[b][i][np.newaxis,:3,...]),step=i)
                #    tf.summary.image("Final PDE trajectory hidden channels, batch "+str(b),np.einsum("ncxy->nxyc",trs_h[b][i][np.newaxis,...]),step=i)
                    
                
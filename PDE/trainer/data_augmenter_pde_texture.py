import jax
import time
import jax.numpy as np
import jax.random as jr
from jaxtyping import Float, Int, PyTree, Scalar, Array
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract

""" 
Use noise as initial condition, learn how to generate textures via LPIPS distance to images.
No fancy intermediate time step stuff, just learn textures as fixed points of the PDE.

"""

class DataAugmenter(DataAugmenterAbstract):
    """
        Inherits the methods of DataAugmenter, but overwrites the batch cloning in the init
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.OVERWRITE_OBS_CHANNELS = False
    def data_init(self,SHARDING=None):
        return None
    
    
    # def sub_trajectory_split(self,L,key=jax.random.PRNGKey(int(time.time()))):
    #     """
    #     Splits data into x (initial conditions) and y (following sub trajectory of length L).
    #     So that x[:,i]->y[:,i+1:i+1+L] is learned

    #     Parameters
    #     ----------
    #     L : Int
    #         Length of subdirectory

    #     Returns
    #     -------
    #     x : float32[BATCHES,CHANNELS,WIDTH,HEIGHT]
    #         Initial conditions
    #     y : float32[BATCHES,L,CHANNELS,WIDTH,HEIGHT]
    #         Following sub trajectories

    #     """
        
        
    #     #pos = list(jax.random.randint(key,shape=(len(self.data_saved),),minval=0,maxval=self.data_true[0].shape[0]-L-1))
    #     #x = jax.tree_util.tree_map(lambda data,p:data[p],self.data_saved,pos)
    #     #y = jax.tree_util.tree_map(lambda data,p:data[p+1:p+1+L],self.data_saved,pos)

        
        
    def data_callback(self,
                    x: PyTree[Float[Array, "C W H"]], 
                    y: PyTree[Float[Array, "N C W H"]], 
                    i: Int[Scalar, ""],
                    key):
        
        """ Preserves intermediate hidden channels after each training iteration

        Returns:
            x : PyTree [BATCHES] f32[1,CHANNELS,WIDTH,HEIGHT]
                Initial conditions
            y : PyTree [BATCHES] f32[L,CHANNELS,WIDTH,HEIGHT]
                True trajectories that follow the initial conditions
        """
        data = self.return_saved_data()
        B = len(data)
        N = data[0].shape[0]
        C = data[0].shape[1]
        W = data[0].shape[2]
        H = data[0].shape[3]
        keys = jr.split(key,B)
        x = [jr.uniform(key,shape=(C,W,H)) for key in keys]
        y = data

        return x,y

        
    # def data_load(self,L,key):
    #     data = self.return_saved_data()
        
    #     return x0,y0

        
    
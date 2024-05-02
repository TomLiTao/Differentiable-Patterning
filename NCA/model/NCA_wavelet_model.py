import jax
import jax.numpy as jnp
import equinox as eqx
import time
import einops
#from Common.model.abstract_model import AbstractModel # Inherit model loading and saving
from NCA.model.NCA_model import NCA

class cNCA(NCA):
    layers: list
    KERNEL_STR: list
    KERNEL_SCALE: int
    WAVELET_NUMBER: int
    N_CHANNELS: int
    N_FEATURES: int
    PERIODIC: bool
    FIRE_RATE: float
    N_WIDTH: int
    def __init__(self, N_CHANNELS, KERNEL_STR=["ID","LAP"], KERNEL_SCALE=1,WAVELET_NUMBER=1, ACTIVATION_STR="relu", PERIODIC=True, FIRE_RATE=1.0, key=jax.random.PRNGKey(int(time.time()))):
        """
        

        Parameters
        ----------
        N_CHANNELS : int
            Number of channels for NCA.
        KERNEL_STR : [STR], optional
            List of strings corresponding to convolution kernels. Can include "ID","DIFF","LAP","AV", corresponding to
            identity, derivatives, laplacian and average respectively. The default is ["ID","LAP"].
        ACTIVATION_STR : str, optional
            Decide which activation function to use. The default is "relu".
        PERIODIC : Boolean, optional
            Decide whether to have periodic or fixed boundaries. The default is True.
        FIRE_RATE : float, optional
            Probability that each pixel updates at each timestep. Defuaults to 1, i.e. deterministic update
        key : jax.random.PRNGKey, optional
            Jax random number key. The default is jax.random.PRNGKey(int(time.time())).

        Returns
        -------
        None.

        """
        
        
        key1,key2 = jax.random.split(key,2)
        self.PERIODIC=PERIODIC
        self.N_CHANNELS = N_CHANNELS
        self.FIRE_RATE = FIRE_RATE
        self.KERNEL_STR = KERNEL_STR
        self.N_WIDTH = 1
        self.KERNEL_SCALE = KERNEL_SCALE
        self.WAVELET_NUMBER = WAVELET_NUMBER
        nstds = 2
        WAVELENGTHS = [jnp.pi*8,jnp.pi*4,jnp.pi*2,jnp.pi]
        WAVELENGTHS = WAVELENGTHS[:self.WAVELET_NUMBER]
        def identity(scale,nstds):
              # Number of standard deviation sigma
            i = jnp.zeros((nstds*scale+1,nstds*scale+1))
            i = i.at[nstds*scale//2,nstds*scale//2].set(1)
            return i

        def gabor(sigma, theta, Lambda, psi, gamma,nstds):
            # General odd gabor filter 
            sigma_x = sigma
            sigma_y = float(sigma) / gamma

            # Bounding box
              # Number of standard deviation sigma
            xmax = max(
                abs(nstds * sigma_x * jnp.cos(theta)), abs(nstds * sigma_y * jnp.sin(theta))
            )
            xmax = jnp.ceil(max(1, xmax))
            ymax = max(
                abs(nstds * sigma_x * jnp.sin(theta)), abs(nstds * sigma_y * jnp.cos(theta))
            )
            ymax = jnp.ceil(max(1, ymax))
            xmin = -xmax
            ymin = -ymax
            (y, x) = jnp.meshgrid(jnp.arange(ymin, ymax + 1), jnp.arange(xmin, xmax + 1))

            # Rotation
            x_theta = x * jnp.cos(theta) + y * jnp.sin(theta)
            y_theta = -x * jnp.sin(theta) + y * jnp.cos(theta)

            gb = jnp.exp(
                -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
            ) * jnp.sin(2 * jnp.pi / Lambda * x_theta + psi) 
            return gb / jnp.sum(jnp.abs(gb))


        def gradx(scale,wavelength,nstds):
            # Gabor filter with large wavelength, x direction
            return gabor(0.5*scale,jnp.pi/2,wavelength,0,1,nstds)

        def grady(scale,wavelength,nstds):
            # Gabor filter with large wavelength, y direction
            return gabor(0.5*scale,0,wavelength,0,1,nstds)
        def gaussian(sigma,nstds):
            # Gaussian
            sigma_x = sigma*0.5
            

            # Bounding box
             # Number of standard deviation sigma
            xmax = nstds * sigma_x
            #max(
            #    abs( )
            #)
            xmax = jnp.ceil(max(1, xmax))
            ymax = nstds * sigma_x
            
            ymax = jnp.ceil(max(1, ymax))
            xmin = -xmax
            ymin = -ymax
            (y, x) = jnp.meshgrid(jnp.arange(ymin, ymax + 1), jnp.arange(xmin, xmax + 1))


            gb = jnp.exp(
                -0.5 * (x**2 / sigma_x**2 + y**2 / sigma_x**2)
            )
            return gb / jnp.sum(gb)

        def laplacian(sigma,nstds):
            # Laplacian of gaussian
            sigma_x = sigma*0.5
            

            # Bounding box
             # Number of standard deviation sigma
            xmax = nstds * sigma_x
            #max(
            #    abs( )
            #)
            xmax = jnp.ceil(max(1, xmax))
            ymax = nstds * sigma_x
            
            ymax = jnp.ceil(max(1, ymax))
            xmin = -xmax
            ymin = -ymax
            (y, x) = jnp.meshgrid(jnp.arange(ymin, ymax + 1), jnp.arange(xmin, xmax + 1))


            gb = jnp.exp(
                -0.5 * (x**2 / sigma_x**2 + y**2 / sigma_x**2)
            )
            lgb = (1-4*(x**2+y**2)/(sigma_x**2))*gb
            
            lap = -lgb
            lap = lap - jnp.mean(lap)
            lap = lap/jnp.sum(jnp.abs(lap))
            
            return lap

        
        # Define which convolution kernels to use
        KERNELS = []
        if "ID" in self.KERNEL_STR:
            id = identity(self.KERNEL_SCALE,nstds)
            KERNELS.append(id)
        if "AV" in self.KERNEL_STR:
            av = gaussian(self.KERNEL_SCALE,nstds)
            KERNELS.append(av)
        if "DIFF" in self.KERNEL_STR:
            for W in WAVELENGTHS:
                dx = gradx(self.KERNEL_SCALE,W,nstds)
                dy = grady(self.KERNEL_SCALE,W,nstds)
                KERNELS.append(dx)
                KERNELS.append(dy)
        if "LAP" in self.KERNEL_STR:
            lap = laplacian(self.KERNEL_SCALE,nstds)
            KERNELS.append(lap)
        
        self.N_FEATURES = N_CHANNELS*len(KERNELS)
        KERNELS = jnp.array(KERNELS) # OHW layout
        
        KERNELS = jnp.zeros((self.N_CHANNELS,self.N_FEATURES//self.N_CHANNELS,nstds*self.KERNEL_SCALE+1,nstds*self.KERNEL_SCALE+1)) + KERNELS[jnp.newaxis]# OIHW layout
        
        
        KERNELS = einops.rearrange(KERNELS,"o i h w -> (o i) (1) h w")
        
        # Define which activation function to use
        if ACTIVATION_STR == "relu":
            ACTIVATION = jax.nn.relu
        elif ACTIVATION_STR == "tanh":
            ACTIVATION = jax.nn.tanh
        elif ACTIVATION_STR == "swish":
            ACTIVATION = jax.nn.swish
        elif ACTIVATION_STR == "linear":
            ACTIVATION = lambda x:x
        elif ACTIVATION_STR == "leaky_relu":
            ACTIVATION = jax.nn.leaky_relu
        elif ACTIVATION_STR == "gelu":
            ACTIVATION = jax.nn.gelu
        else:
            ACTIVATION = None
        
        @jax.jit
        def periodic_pad(x):
            if self.PERIODIC:
                return jnp.pad(x, ((0,0),(nstds*self.KERNEL_SCALE//2,nstds*self.KERNEL_SCALE//2),(nstds*self.KERNEL_SCALE//2,nstds*self.KERNEL_SCALE//2)), mode='wrap')
            else:
                return x
        
        @jax.jit
        def periodic_unpad(x):
            if self.PERIODIC:
                return x[:,nstds*self.KERNEL_SCALE//2:-nstds*self.KERNEL_SCALE//2,nstds*self.KERNEL_SCALE//2:-nstds*self.KERNEL_SCALE//2]
            else:
                return x
        

        self.layers = [
            periodic_pad,
            eqx.nn.Conv2d(in_channels=self.N_CHANNELS,
                            out_channels=self.N_FEATURES,
                            kernel_size=nstds*self.KERNEL_SCALE+1,
                            use_bias=False,
                            key=key1,
                            padding=nstds*self.KERNEL_SCALE//2,
                            groups=self.N_CHANNELS),
            periodic_unpad,
            eqx.nn.Conv2d(in_channels=self.N_FEATURES,
                            out_channels=self.N_WIDTH*self.N_FEATURES,
                            kernel_size=1,
                            use_bias=False,
                            key=key1),
            ACTIVATION,
            eqx.nn.Conv2d(in_channels=self.N_WIDTH*self.N_FEATURES, 
                            out_channels=self.N_CHANNELS,
                            kernel_size=1,
                            use_bias=True,
                            key=key2)
            ]
        
        
        # Initialise final layer to zero
        w_zeros = jnp.zeros((self.N_CHANNELS,self.N_WIDTH*self.N_FEATURES,1,1))
        b_zeros = jnp.zeros((self.N_CHANNELS,1,1))
        w_where = lambda l: l.weight
        b_where = lambda l: l.bias
        self.layers[-1] = eqx.tree_at(w_where,self.layers[-1],w_zeros)
        self.layers[-1] = eqx.tree_at(b_where,self.layers[-1],b_zeros)
        
        # Initialise first layer weights as perception kernels
        self.layers[1] = eqx.tree_at(w_where,self.layers[1],KERNELS)


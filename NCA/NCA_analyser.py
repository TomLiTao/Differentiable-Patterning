import jax
import jax.numpy as np
import equinox as eqx
from einops import rearrange
import optimistix as optx


# Methods for doing gradient based analysis of trained NCA models




@eqx.filter_jit
def kernel_channel_sensitivity_per_pixel(nca,x,KERNEL,in_channel,out_channel):
    # Computes spatial sensitivity between input and output channel for a given kernel
    # i.e. how much does each pixel of the output channel change when the input channel is changed, for a given kernel and input state
    CHANNELS = nca.N_CHANNELS
    y = nca.perception(x)
    #print(y.shape)
    grad_param = y[CHANNELS*KERNEL:(KERNEL+1)*CHANNELS][in_channel]
    
    aux = (y,nca,out_channel)
    def f(grad_param):
        y,nca,out_channel = aux
        y = y.at[CHANNELS*KERNEL:(KERNEL+1)*CHANNELS].set(grad_param)
        h = nca.layers[0](y)
        internal_activations = nca.layers[1](h)
        updates = nca.layers[2](internal_activations)
        return updates[out_channel]
    output,G = jax.vjp(f,grad_param)
    return output,np.array(G(np.ones(output.shape)))



@eqx.filter_jit
def full_sensitivity_vmap(nca,x):
    # Iterate kernel_channel_sensitivity_per_pixel over all input channels, output channels and kernels
    v_kernel_channel_sensitivity = jax.vmap(kernel_channel_sensitivity_per_pixel,in_axes=(None,None,None,None,0),out_axes=(0,0))
    
    dupdates = []
    for i in range(nca.N_CHANNELS): # Loop over input channels
        dd = []
        for k in range(4): # Loop over kernels
            _,d = v_kernel_channel_sensitivity(nca,x,k,i,np.arange(32))
            dd.append(d)

        dupdates.append(dd)

    dupdates = np.array(dupdates)
    dupdates = rearrange(dupdates,"In K Out () x y -> K In Out x y")
    return dupdates

@eqx.filter_jit
def nca_fixed_point(nca,x_guess,key,rtol=1e-1,atol=1e-1):
    

    func = lambda x,args: nca(x,lambda x:x,key=key)
    solver = optx.LevenbergMarquardt(rtol=rtol,atol=atol)
    return optx.fixed_point(func,solver,y0=x_guess)
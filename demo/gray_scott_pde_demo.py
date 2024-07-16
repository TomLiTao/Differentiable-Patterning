import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.path.append('..')
from PDE.model.fixed_models.update_gray_scott import F as F_gray_scott
from PDE.model.solver.semidiscrete_solver import PDE_solver
from Common.model.spatial_operators import Ops
from Common.utils import my_animate


SIZE = 64 # Grid size




key = jr.PRNGKey(int(time.time())) # JAX PRNG key generation
x0 = jr.uniform(key,shape=(2,SIZE,SIZE))


# Make smoothly random initial conditions
op = Ops(PADDING="CIRCULAR",dx=1.0,KERNEL_SCALE=3)
for i in range(5):
    x0 = op.Average(x0)
x0 = x0.at[0].set(jnp.where(x0[0]>0.51,0.0,1.0))
x0 = op.Average(x0)
x0 = x0.at[1].set(1-x0[0])



# Define RHS of PDE
func = F_gray_scott(PADDING="CIRCULAR",    # Boundary condition, choose from PERIODIC, ZEROS, REFLECT, REPLICATE
                    dx=0.5,                # Spatial stepsize - scales spatial patterns, but also affects stability
                    alpha=0.0623,          # Parameters that affect patterning 
                    gamma=0.06268)         # 
solver = PDE_solver(func,dt=0.2)           # Wrap the RHS in a numerical solver
T,Y = solver(ts=jnp.linspace(0,10000,100),y0=x0) # Generate solution trajectory

print(T.shape)
print(Y.shape)
my_animate(Y[:,:1],clip=False)

# Save the generated images and create the animation
def save_images_and_animate(img, save_path='frames', anim_path='animation.mp4', clip=True):
    """
    Save each frame as an image and create an animation.
    Parameters
    ----------
    img : float32 or int array [N,rgb,_,_]
        img must be float in range [0,1] 
    save_path : str
        Directory to save the individual frames.
    anim_path : str
        Path to save the animation file.
    clip : bool
        Whether to clip the image values to [0,1].
    """
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if clip:
        im_min = 0
        im_max = 1
        img = np.clip(img, im_min, im_max)
    else:
        im_min = np.min(img)
        im_max = np.max(img)
    
    img = np.einsum("ncxy->nxyc", img)
    frames = []  # for storing the generated images
    fig = plt.figure()
    
    for i in range(img.shape[0]):
        frame_path = os.path.join(save_path, f'frame_{i:04d}.png')
        plt.imshow(img[i], vmin=im_min, vmax=im_max)
        plt.savefig(frame_path)
        frames.append([plt.imshow(img[i], vmin=im_min, vmax=im_max, animated=True)])
        plt.close()

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=0)
    ani.save(anim_path)
    plt.show()

save_images_and_animate(Y[:, :1], clip=False)

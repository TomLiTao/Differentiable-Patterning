import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import io



def plot_to_image(figure):
	"""Converts the matplotlib plot specified by 'figure' to a PNG image and
	returns it. The supplied figure is closed and inaccessible after this call."""
	# Save the plot to a PNG in memory.
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	# Closing the figure prevents it from being displayed directly inside
	# the notebook.
	plt.close(figure)
	buf.seek(0)
	# Convert PNG buffer to TF image
	image = tf.image.decode_png(buf.getvalue(), channels=4)
	# Add the batch dimension
	image = tf.expand_dims(image, 0)
	return image


def plot_weight_matrices(pde):
    """
    Plots heatmaps of NCA layer weights

    Parameters
    ----------
    pde : object callable - (float32 array [T], float32 array [N_CHANNELS,_,_]) -> (float32 array [T,N_CHANNELS,_,_])
        the PDE solver object to plot parameters of

    Returns
    -------
    figs : list of images
        a list of images

    """
    w_cell_motility = pde.func.cell_motility.motility_layers[-2].weight[:,:,0,0]
    w_cell_chemotaxis = pde.func.cell_motility.chemotaxis_layers[-2].weight[:,:,0,0]

    w_cell_reaction = pde.func.cell_reaction.layers[-2].weight[:,:,0,0]

    w_signal_diffusion = pde.func.signal_diffusion.diffusion_constants.weight[:,:,0,0]

    w_signal_reaction_decay = pde.func.signal_reaction.decay_layers[-2].weight[:,:,0,0]
    w_signal_reaction_production = pde.func.signal_reaction.production_layers[-2].weight[:,:,0,0]
    #w2_v = pde.func.f_v.layers[2].weight[:,:,0,0]

    #w1_d = pde.func.f_d.layers[-1].weight[:,:,0,0]

    #w1_r = pde.func.f_r.layers[0].weight[:,:,0,0]
    #w2_r = pde.func.f_r.layers[2].weight[:,:,0,0]
    figs = []

    figure = plt.figure(figsize=(5,5))
    col_range = max(np.max(w_cell_motility),-np.min(w_cell_motility))
    plt.imshow(w_cell_motility,cmap="seismic",vmax=col_range,vmin=-col_range)
    plt.ylabel("Output")
    plt.xlabel("Input")
    plt.title("Cell Motility Final Layer")
    figs.append(plot_to_image(figure))

    figure = plt.figure(figsize=(5,5))
    col_range = max(np.max(w_cell_chemotaxis),-np.min(w_cell_chemotaxis))
    plt.imshow(w_cell_chemotaxis,cmap="seismic",vmax=col_range,vmin=-col_range)
    plt.ylabel("Output")
    plt.xlabel("Input")
    plt.title("Cell Chemotaxis Final Layer")
    figs.append(plot_to_image(figure))

    figure = plt.figure(figsize=(5,5))
    #col_range = max(np.max(w1_d),-np.min(w1_d))
    #plt.imshow(w1_d,cmap="seismic",vmax=col_range,vmin=-col_range)
    plt.plot(w_signal_diffusion)
    plt.ylabel("Weight")
    plt.xlabel("Channel")
    plt.title("Signal Diffusion weights")
    figs.append(plot_to_image(figure))

    figure = plt.figure(figsize=(5,5))
    col_range = max(np.max(w_cell_reaction),-np.min(w_cell_reaction))
    plt.imshow(w_cell_reaction,cmap="seismic",vmax=col_range,vmin=-col_range)
    plt.ylabel("Output")
    plt.xlabel("Input")
    plt.title("Cell Proliferation Final Layer")
    figs.append(plot_to_image(figure))

    figure = plt.figure(figsize=(5,5))
    col_range = max(np.max(w_signal_reaction_decay),-np.min(w_signal_reaction_decay))
    plt.imshow(w_signal_reaction_decay,cmap="seismic",vmax=col_range,vmin=-col_range)
    plt.ylabel("Output")
    plt.xlabel("Input")
    plt.title("Signal Decay Final Layer")
    figs.append(plot_to_image(figure))


    figure = plt.figure(figsize=(5,5))
    col_range = max(np.max(w_signal_reaction_production),-np.min(w_signal_reaction_production))
    plt.imshow(w_signal_reaction_production,cmap="seismic",vmax=col_range,vmin=-col_range)
    plt.ylabel("Output")
    plt.xlabel("Input")
    plt.title("Signal Production Final Layer")
    figs.append(plot_to_image(figure))

    return figs

def plot_weight_kernel_boxplot(pde):
    """
    Plots boxplots of PDE 1st layer weights sorted by which channel they correspond to

    Parameters
    ----------
    pde : object callable - (float32 array [T], float32 array [N_CHANNELS,_,_]) -> (float32 array [T,N_CHANNELS,_,_])
        the PDE solver object to plot parameters of

    Returns
    -------
    figs : list of images
        a list of images

    """


    w_cell_motility = pde.func.cell_motility.motility_layers[0].weight[:,:,0,0]
    w_cell_chemotaxis = pde.func.cell_motility.chemotaxis_layers[0].weight[:,:,0,0]

    w_cell_reaction = pde.func.cell_reaction.layers[0].weight[:,:,0,0]

    w_signal_reaction_decay = pde.func.signal_reaction.decay_layers[0].weight[:,:,0,0]
    w_signal_reaction_production = pde.func.signal_reaction.production_layers[0].weight[:,:,0,0]
    
    

    figs = []

    figure = plt.figure(figsize=(5,5))
    plt.boxplot(w_cell_motility.T)
    plt.xlabel("Channels")
    plt.ylabel("Weights")
    plt.title("Cell Motility 1st layer")
    figs.append(plot_to_image(figure))

    figure = plt.figure(figsize=(5,5))
    plt.boxplot(w_cell_chemotaxis.T)
    plt.xlabel("Channels")
    plt.ylabel("Weights")
    plt.title("Cell Chemotaxis 1st layer")
    figs.append(plot_to_image(figure))

    figure = plt.figure(figsize=(5,5))
    plt.boxplot(w_cell_reaction.T)
    plt.xlabel("Channels")
    plt.ylabel("Weights")
    plt.title("Cell Proliferation 1st layer")
    figs.append(plot_to_image(figure))

    figure = plt.figure(figsize=(5,5))
    plt.boxplot(w_signal_reaction_production.T)
    plt.xlabel("Channels")
    plt.ylabel("Weights")
    plt.title("Signal Production 1st layer")
    figs.append(plot_to_image(figure))

    figure = plt.figure(figsize=(5,5))
    plt.boxplot(w_signal_reaction_decay.T)
    plt.xlabel("Channels")
    plt.ylabel("Weights")
    plt.title("Signal Decay 1st layer")
    figs.append(plot_to_image(figure))

    return figs

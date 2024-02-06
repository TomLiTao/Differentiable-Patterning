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

def plot_weight_matrices(model):
	w_0 = model.Agent_nn.layers[1].weight
   
	w_ph = model.Agent_nn.layers_pheremone[0].weight
	figs = []

	figure = plt.figure(figsize=(5,5))
	col_range = max(np.max(w_0),-np.min(w_0))
	plt.imshow(w_0,cmap="seismic",vmax=col_range,vmin=-col_range)
	plt.ylabel("Output")
	plt.xlabel(r"Observations")
	figs.append(plot_to_image(figure))


	figure = plt.figure(figsize=(5,5))
	col_range = max(np.max(w_ph),-np.min(w_ph))
	plt.imshow(w_ph,cmap="seismic",vmax=col_range,vmin=-col_range)
	plt.xlabel("Input from previous layer")
	plt.ylabel("Pheremone lattice increments")
	figs.append(plot_to_image(figure))
	return figs

def scatter_wrap(X,x_label=None,y_label=None):
	x = X[...,0]
	y = X[...,1]
	fig = plt.figure(figsize=(5,5))
	l_max = np.max(np.array([x,y]))
	plt.scatter(x,y)
	plt.xlim(0,l_max)
	plt.ylim(0,l_max)
	if x_label ==None:
		plt.xlabel("x")
	else:
		plt.xlabel(x_label)
	if y_label ==None:
		plt.ylabel("y")
	else:
		plt.ylabel(y_label)
	return plot_to_image(fig)


def parallel_scatter_wrap(X,x_label=None,y_label=None):

	x = X[...,0,:]
	y = X[...,1,:]
	B = x.shape[0]
	figs = []
	for b in range(B):
		fig = plt.figure(figsize=(5,5))
		l_max = np.max(np.array([x[b],y[b]]))
		plt.scatter(x[b],y[b],s=0.1)
		plt.xlim(0,l_max)
		plt.ylim(0,l_max)
		if x_label ==None:
			plt.xlabel("x")
		else:
			plt.xlabel(x_label)
		if y_label ==None:
			plt.ylabel("y")
		else:
			plt.ylabel(y_label)
		ar = plot_to_image(fig)[0]
		figs.append(ar)
	return figs
	
def my_animate_agents(img,agents):
	"""
	Produces animation of agents on a scalar field background
	Parameters
	----------
	img : float32 or int array [N,rgb,_,_]
		img must be float in range [0,1] 
	agents : float32 array [N,type,2,M]
        X and Y coordinates of agent locations
		type indexes whether position (0) or velocity (1)
	"""
	img = np.clip(img,0,1)
	img = np.einsum("ncxy->nxyc",img)
	frames = [] # for storing the generated images
	fig = plt.figure()
	agents = np.array(agents)
	print(agents.shape)
	#agents = agents.reshape((-1,2))
	for i in range(img.shape[0]):
		frames.append([plt.imshow(img[i,:,:,:3],vmin=0,vmax=1,animated=True,origin="lower"),
				       plt.scatter(agents[i,0,1],agents[i,0,0],animated=True,color="white",s=0.5)])
		#;frames.append([plt.scatter(agents[i,0,0],agents[i,0,1],animated=True)])
	ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,repeat_delay=0)
	plt.show()
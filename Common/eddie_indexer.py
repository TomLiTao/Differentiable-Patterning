import numpy as np
from Common.trainer.loss import l2,euclidean,vgg,spectral,random_sampled_euclidean




"""
    Functions that take in the Eddie job array index and return a tuple of training hyperparameters
"""

def index_to_learnrate_parameters(index):
	"""
		Takes job array index from 1-84 and constructs 4-tuple of learn rate, optimiser, training mode and gradient normalisation

	"""
	learn_rates = np.logspace(1,-5,7)
	learn_rates_name = ["1e1","1e0","1e-1","1e-2","1e-3","1e-4","1e-5"]
	optimisers = ["Adagrad",
				  "Adam",
				  "Nadam"]
	training_modes = ["differential","full"]
	grad_norm = [True,False]
	L1 = len(learn_rates)
	L2 = len(optimisers)
	L3 = len(training_modes)
	L4 = len(grad_norm)

	indices = np.unravel_index(index,(L1,L2,L3,L4))
	lr= learn_rates[indices[0]]
	lr_name = learn_rates_name[indices[0]]
	opt = optimisers[indices[1]]
	mode = training_modes[indices[2]]
	grad = grad_norm[indices[3]]
	return lr,lr_name,opt,mode,grad


def index_to_Nadam_parameters(index):
	"""
		Takes index from 1-N and constructs 3-tuple of learn rate, NCA to PDE step ratio and grad_norm

	"""
	learn_rates = np.logspace(1,-5,7)
	learn_rates_name = ["1e1","1e0","1e-1","1e-2","1e-3","1e-4","1e-5"]
	ratios = [1,2,4,8]
	grad_norm = [True,False]
	L1 = len(learn_rates)
	L2 = len(ratios)
	L3 = len(grad_norm)
	indices = np.unravel_index(index,(L1,L2,L3))
	lr = learn_rates[indices[0]]
	lr_name = learn_rates_name[indices[0]]
	ratio = ratios[indices[1]]
	grad = grad_norm[indices[2]]
	return lr,lr_name,ratio,grad



def index_to_mitosis_parameters(index):
	"""
		Takes index from 1-N and constructs 2-tuple of loss function and time sampling rate 
	"""
	loss_funcs = [loss_spectral,
				  loss_bhattacharyya_modified,
				  loss_hellinger_modified,
				  None]
	loss_func_name = ["spectral","bhattachryya","hellinger","euclidean"]
	sampling_rates = [1,2,4,6,8,12,16,24,32]
	instance = [0,1,2,3]
	L1 = len(loss_funcs)
	L2 = len(sampling_rates)
	L3 = len(instance)
	indices = np.unravel_index(index,(L1,L2,L3))
	loss = loss_funcs[indices[0]]
	loss_name = loss_func_name[indices[0]]
	sampling = sampling_rates[indices[1]]
	i = instance[indices[2]]
	return loss,loss_name,sampling,i

def index_to_generalise_test(index):
    loss_funcs = [None,
                  loss_bhattacharyya_modified,
                  loss_bhattacharyya_euclidean,
				  loss_spectral,
				  loss_hellinger_modified,
				  loss_kl_divergence]
    loss_func_names = ["euclidean",
                       "bhattacharyya",
                       "bhattacharyya_euclidean",
					   "spectral",
					   "hellinger",
					   "kl_divergence"]
    sampling_rates = [1,2,4,8,16]
    tasks = ["heat",
            "mitosis",
            "coral",
			"gol"]
    #sampling_rates = [16,32,48,64]
    #tasks=["emoji"]
    L1 = len(loss_funcs)
    L2 = len(sampling_rates)
    L3 = len(tasks)
    indices = np.unravel_index(index,(L1,L2,L3))
    loss = loss_funcs[indices[0]]
    loss_name = loss_func_names[indices[0]]
    sampling = sampling_rates[indices[1]]
    task = tasks[indices[2]]
    return loss,loss_name,sampling,task

def index_to_generalise_test_2():
	#Rerunning some of index_tp_generalise_test, as fft loss was bugged
	loss = loss_spectral
	loss_name = "spectral"
	sampling = 8
	task = "coral"
	return loss,loss_name,sampling,task
def index_to_model_exploration_parameters(index):
    loss_funcs = [None,loss_spectral]
    loss_func_names = ["euclidean","spectral"]
    tasks = ["heat","mitosis","coral","emoji"]
    layers = [2,3]
    kernels = ["ID_LAP","ID_LAP_AV","ID_DIFF_LAP","ID_DIFF"]
    activations=["linear","relu","swish","tanh"]
    L1 = len(loss_funcs)
    L2 = len(tasks)
    L3 = len(layers)
    L4 = len(kernels)
    L5 = len(activations)
    indices = np.unravel_index(index, (L1,L2,L3,L4,L5))
    LOSS = loss_funcs[indices[0]]
    LOSS_STR = loss_func_names[indices[0]]
    TASK = tasks[indices[1]]
    LAYER = layers[indices[2]]
    KERNEL= kernels[indices[3]]
    ACTIVATION= activations[indices[4]]
    return LOSS,LOSS_STR,TASK,LAYER,KERNEL,ACTIVATION
    

def index_to_activations_and_kernels(index):
	
	
	
	
	
	kernels = [["ID","LAP"],
			   ["ID","LAP","AV"],
			   ["ID","DIFF","LAP"],
			   ["ID","DIFF"]]
	activations=["linear","relu","swish","tanh","leaky_relu","gelu"]
	
	L1 = len(kernels)
	L2 = len(activations)
	indices = np.unravel_index(index, (L1,L2))
	
	KERNEL= kernels[indices[0]]
	ACTIVATION= activations[indices[1]]
	return KERNEL,ACTIVATION

def index_to_emoji_symmetry_parameters(index):
	kernels = ["ID_LAP_AV","ID_DIFF_LAP"]
	tasks=["normal","rotated_data","rotation_task","translation_task"]
	L1 = len(kernels)
	L2 = len(tasks)
	indices = np.unravel_index(index,(L1,L2))
	KERNEL = kernels[indices[0]]
	TASK = tasks[indices[1]]
	return KERNEL,TASK
	
def index_to_channel_sample(index):
	channels = [4,6,8,10,12,14,16,32]
	samplings = [1,2,4,8,16,32,64]
	L1 = len(channels)
	L2 = len(samplings)
	indices = np.unravel_index(index,(L1,L2))
	N_CHANNELS = channels[indices[0]]
	SAMPLING = samplings[indices[1]]
	return N_CHANNELS,SAMPLING


def index_to_channel(index):
	channels = [4,6,8,10,12,14,16,32]
	SAMPLING=64
	N_CHANNELS = channels[index]
	return N_CHANNELS,SAMPLING
	

def index_to_sample(index):
	N_CHANNELS = 16
	samplings = [1,2,4,8,16,32,128]
	SAMPLING=samplings[index]
	return N_CHANNELS,SAMPLING


def index_to_data_nca_type(index):
	indices = np.unravel_index(index,(8,2))
	data_index = indices[0]
	nca_type_index = indices[1]
	return data_index,nca_type_index
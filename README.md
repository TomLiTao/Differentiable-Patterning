# Differentiable-Patterning
A collection of different projects and ideas that use differentiable programming to explore self organising systems with emergent pattern formation. Primarily for my PhD research

## Requirements 
 - tensorflow 2.13.0 (just for tensorboard logging)
 - tensorboard 2.13.0
 - numpy 1.24.4
 - scipy 1.9.0
 - scikit-image 0.19.1
 - tqdm 4.64.0
 - matplotlib 3.7.2
 - jax 0.4.13
 - jaxlib 0.4.13
 - optax 0.1.7
 - equinox 0.10.4


## Code Structure

- Neural Cellular Automata (NCA)
  - NCA/model/NCA_model.py includes a JAX/Equinox implementation of the NCA model
  - NCA/model/boundary.py includes handling of complex boundary conditions and hard coded environment channels
  - NCA/trainer/NCA_trainer.py includes a class that uses Optax to fit the NCA models to data
  - NCA/trainer/data_augmenter.py includes a class for augmenting data during and before training. Handles explicit multi-gpu parallelism as well.
  - NCA/trainer/data_augmenter_tree.py performs the same data augmentation as above, but on PyTrees of data, to allow training simultaneously on multiple trajectories of different sizes
  - NCA/trainer/loss.py includes loss function definitions
  - NCA/trainer/tensorboard_log.py logs performance of model during training to tensorboard
  - NCA/NCA_visualiser.py includes methods for visualising and interpretting trained NCA
  - NCA/utils.py includes various file io and data pre-processing methods
- Partial Differential Equations (PDE)
  - PDE/reaction_diffusion_advection/update.py includes a spatially discretised version of a general multi-species reaction diffusion advection equation, as an Equinox module
  - PDE/solver/semi_discrete_solver.py includes an auto-differentiable numerical solver of spatially discretised PDEs, implemented in Diffrax
  - PDE/trainer/data_augmenter_pde.py includes a subclass of NCA/trainer/data_augmenter_tree, tailored to PDE training
  - PDE/trainer/PDE_trainer.py includes a class that uses Optax to fit PDE paramaters such that the solutions of the PDE approximate a given time series
  - PDE/trainer/optimiser.py includes a custom Optax optimiser that keeps diffusion coefficients non-negative
  - PDE/trainer/tensorboard_log.py logs performance of model during training to tensorboard
  - PDE/PDE_visualiser.py includes methods for visualising and interpretting trained PDE

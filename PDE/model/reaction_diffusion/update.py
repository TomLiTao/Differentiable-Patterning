#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:46:20 2023

@author: s1605376
"""
import jax
import equinox as eqx
import jax.numpy as jnp
import time

from PDE.model.reaction_diffusion.reaction import R
from PDE.model.reaction_diffusion.diffusion_nonlinear import D
from jaxtyping import Array, Float, PyTree, Scalar

class F(eqx.Module):
	f_r: R
	f_d: D
	N_CHANNELS: int
	N_LAYERS: int
	ORDER: int
	PADDING: str
	dx: float
	def __init__(self,
			  	 N_CHANNELS,
				 PADDING,
				 dx,
				 INTERNAL_ACTIVATION=jax.nn.relu,
				 INIT_SCALE={"reaction":0.5,"diffusion":0.5},
				 INIT_TYPE={"reaction":"normal","diffusion":"normal"},
				 USE_BIAS=True,
				 STABILITY_FACTOR=0.01,
				 ORDER=1,
				 N_LAYERS=1,
				 ZERO_INIT={"reaction":True,"diffusion":False},
				 key=jax.random.PRNGKey(int(time.time()))):
		
		
		self.N_CHANNELS = N_CHANNELS
		self.PADDING = PADDING
		self.dx = dx
		self.N_LAYERS = N_LAYERS
		self.ORDER = ORDER
		keys = jax.random.split(key,2)

		self.f_r = R(N_CHANNELS=N_CHANNELS,
			   		 INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
					 OUTER_ACTIVATION=jax.nn.relu, # SHOULD BE STRICTLY NON NEGATIVE FOR INTERPRETABILITY
					 INIT_SCALE=INIT_SCALE["reaction"], # Should be much smaller initial scaling
					 INIT_TYPE=INIT_TYPE["reaction"],
					 USE_BIAS=USE_BIAS,
					 STABILITY_FACTOR=STABILITY_FACTOR,
					 ORDER=ORDER,
					 N_LAYERS=N_LAYERS,
					 ZERO_INIT=ZERO_INIT["reaction"],
					 key=keys[0])

		self.f_d = D(N_CHANNELS=N_CHANNELS,
			   		 PADDING=PADDING,
					 dx=dx,
			   		 INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
			   		 OUTER_ACTIVATION=jax.nn.relu, # MUST BE STRICTLY NON NEGATIVE FOR NUMERICAL STABILITY
					 INIT_SCALE=INIT_SCALE["diffusion"],
					 INIT_TYPE=INIT_TYPE["diffusion"],
					 USE_BIAS=USE_BIAS,
					 ORDER=ORDER,
					 N_LAYERS=N_LAYERS,
					 ZERO_INIT=ZERO_INIT["diffusion"],
					 key=keys[1])

	@eqx.filter_jit
	def __call__(self,
			  	 t: Float[Scalar, ""],
				 X: Float[Array, "{self.N_CHANNELS} x y"],
				 args)-> Float[Array, "{self.N_CHANNELS} x y"]:
		"""

		Parameters
		----------
		t : float32
			timestep
		
		X : float32 [N_CHANNELS,_,_]
			input PDE lattice state.
		
		args : None
			Required for format of diffrax.ODETerm

		Returns
		-------
		X_update : float32 [N_CHANNELS,_,_]
			update to PDE lattice state state.

		"""
		return self.f_d(X) + self.f_r(X)
	
	def partition(self):
		r_diff,r_static = self.f_r.partition()
		d_diff,d_static = self.f_d.partition()
		total_diff,total_static = eqx.partition(self,eqx.is_array)
		#print(total_diff)
		#print(total_static)
		where_r = lambda m:m.f_r
		where_d = lambda m:m.f_d
		
		total_diff = eqx.tree_at(where_r,total_diff,r_diff)
		total_diff = eqx.tree_at(where_d,total_diff,d_diff)
		
		total_static = eqx.tree_at(where_r,total_static,r_static)
		total_static = eqx.tree_at(where_d,total_static,d_static)
		return total_diff,total_static
	
	def combine(self,diff,static):
		self = eqx.combine(diff,static)
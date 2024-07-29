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
from PDE.model.reaction_diffusion_advection.advection import V
from PDE.model.reaction_diffusion_advection.reaction import R
from PDE.model.reaction_diffusion_advection.diffusion_nonlinear import D
from jaxtyping import Array, Float, PyTree, Scalar

class F(eqx.Module):
	f_v: V
	f_r: R
	f_d: D
	N_CHANNELS: int
	PADDING: str
	dx: float
	def __init__(self,
			  	 N_CHANNELS,
				 PADDING,
				 dx,
				 INTERNAL_ACTIVATION=jax.nn.relu,
				 ADVECTION_OUTER_ACTIVATION=jax.nn.tanh,
				 INIT_SCALE=0.01,
				 USE_BIAS=True,
				 STABILITY_FACTOR=0.01,
				 REACTION_INIT_SCALE_RATIO=0.01,
				 ZERO_INIT=True,
				 key=jax.random.PRNGKey(int(time.time()))):
		self.N_CHANNELS = N_CHANNELS
		self.PADDING = PADDING
		self.dx = dx
		key1,key2,key3 = jax.random.split(key,3)

		self.f_r = R(N_CHANNELS=N_CHANNELS,
			   		 INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
					 OUTER_ACTIVATION=jax.nn.relu, # SHOULD BE STRICTLY NON NEGATIVE FOR INTERPRETABILITY
					 INIT_SCALE=INIT_SCALE*REACTION_INIT_SCALE_RATIO, # Should be much smaller initial scaling
					 USE_BIAS=USE_BIAS,
					 STABILITY_FACTOR=STABILITY_FACTOR,
					 ZERO_INIT=ZERO_INIT,
					 key=key1)
		self.f_v = V(N_CHANNELS=N_CHANNELS,
			   		 PADDING=PADDING,
					 dx=dx,
			   		 INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
			   		 OUTER_ACTIVATION=ADVECTION_OUTER_ACTIVATION, # Can be any activation function
					 INIT_SCALE=INIT_SCALE,
					 USE_BIAS=USE_BIAS,
					 ZERO_INIT=ZERO_INIT,
					 DIM = 2,																
			   		 key=key2)
		self.f_d = D(N_CHANNELS=N_CHANNELS,
			   		 PADDING=PADDING,
					 dx=dx,
			   		 INTERNAL_ACTIVATION=INTERNAL_ACTIVATION,
			   		 OUTER_ACTIVATION=jax.nn.relu, # MUST BE STRICTLY NON NEGATIVE FOR NUMERICAL STABILITY
					 INIT_SCALE=INIT_SCALE,
					 USE_BIAS=USE_BIAS,
					 ZERO_INIT=False,
					 key=key3)

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
		return self.f_d(X) - self.f_v(X) + self.f_r(X)
	
	def partition(self):
		r_diff,r_static = self.f_r.partition()
		v_diff,v_static = self.f_v.partition()
		d_diff,d_static = self.f_d.partition()
		total_diff,total_static = eqx.partition(self,eqx.is_array)
		#print(total_diff)
		#print(total_static)
		where_r = lambda m:m.f_r
		where_v = lambda m:m.f_v
		where_d = lambda m:m.f_d
		
		total_diff = eqx.tree_at(where_r,total_diff,r_diff)
		total_diff = eqx.tree_at(where_v,total_diff,v_diff)
		total_diff = eqx.tree_at(where_d,total_diff,d_diff)
		
		total_static = eqx.tree_at(where_r,total_static,r_static)
		total_static = eqx.tree_at(where_v,total_static,v_static)
		total_static = eqx.tree_at(where_d,total_static,d_static)
		return total_diff,total_static
	
	def combine(self,diff,static):
		self = eqx.combine(diff,static)
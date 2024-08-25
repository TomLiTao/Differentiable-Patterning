import optax
import jax
import equinox as eqx


def non_negative_diffusion(schedule,optimiser=optax.nadamw):

	opt_ra = optax.chain(optax.scale_by_param_block_norm(),
					  	 optimiser(schedule)) # Adam with weight decay for reaction and advection
	opt_d = optax.chain(optax.keep_params_nonnegative(),
					 	optax.scale_by_param_block_norm(),
						optax.nadam(schedule)) # Non-negative adam on diffusive terms (no weight decay)
	
	def label_diffusive(tree):
		# Returns True for the diffusion terms that should remain non-negative
		
		filter_spec = jax.tree_util.tree_map(lambda _:False,tree)
		filter_spec = eqx.tree_at(lambda t:t.func.f_d.diffusion_constants.weight,filter_spec,replace=True)
		return filter_spec
	
	def label_not_diffusive(tree):
		# Returns True for the parameters that are NOT the non-negative diffusive terms
		filter_spec = jax.tree_util.tree_map(lambda _:True,tree)
		filter_spec = eqx.tree_at(lambda t:t.func.f_d.diffusion_constants.weight,filter_spec,replace=False)
		return filter_spec
	
	opt_ra = optax.masked(opt_ra,label_not_diffusive)
	opt_d = optax.masked(opt_d,label_diffusive)
	
	
	return optax.chain(opt_d,opt_ra)


def multi_learnrate(schedule,rate_ratios,optimiser=optax.nadam,pre_process=optax.identity()):
	
	schedule_funcs = [
		#lambda s:schedule(s)*rate_ratios["advection"],
		lambda s:schedule(s)*rate_ratios["reaction"],
		lambda s:schedule(s)*rate_ratios["diffusion"]
	]
	
	def label_r(tree):
		# Returns True for the reaction terms
		filter_spec = jax.tree_util.tree_map(lambda _:False,tree)
		filter_spec = eqx.tree_at(lambda t:t.func.f_r.production_layers,filter_spec,replace=True)
		filter_spec = eqx.tree_at(lambda t:t.func.f_r.decay_layers,filter_spec,replace=True)
		return filter_spec
	
	# def label_a(tree):
	# 	# Returns True for the advection terms
	# 	filter_spec = jax.tree_util.tree_map(lambda _:False,tree)
	# 	filter_spec = eqx.tree_at(lambda t:t.func.f_v.layers,filter_spec,replace=True)
	# 	return filter_spec
	
	def label_d(tree):
		# Returns True for the diffusion terms
		filter_spec = jax.tree_util.tree_map(lambda _:False,tree)
		filter_spec = eqx.tree_at(lambda t:t.func.f_d.layers,filter_spec,replace=True)
		return filter_spec
	
	#opt_a = optax.masked(optax.chain(pre_process,optax.apply_if_finite(optimiser(schedule_funcs[0]),max_consecutive_errors=8)),label_a)
	opt_r = optax.masked(optax.chain(pre_process,optax.apply_if_finite(optimiser(schedule_funcs[0]),max_consecutive_errors=8)),label_r)
	opt_d = optax.masked(optax.chain(pre_process,optax.apply_if_finite(optimiser(schedule_funcs[1]),max_consecutive_errors=8)),label_d)

	#return optax.chain(opt_a,opt_r,opt_d)
	return optax.chain(opt_r,opt_d)
	#param_labels = ("advection","reaction","diffusion")
	# opt = optax.multi_transform(
	# 	{"advection":opt_a,
	# 	 "reaction":opt_r,
	# 	 "diffusion":opt_d},
	# 	 param_labels)
	
	# return opt

def non_negative_diffusion_chemotaxis(schedule,optimiser=optax.nadamw):
	
	opt_ra = optax.chain(optax.scale_by_param_block_norm(),
					  	 optimiser(schedule)) # Adam with weight decay for reaction and advection
	opt_d = optax.chain(optax.keep_params_nonnegative(),
					    optax.scale_by_param_block_norm(),
						optax.nadam(schedule)) # Non-negative adam on diffusive terms (no weight decay)
	
	def label_diffusive(tree):
		# Returns True for the diffusion terms that should remain non-negative
		
		filter_spec = jax.tree_util.tree_map(lambda _:False,tree)
		filter_spec = eqx.tree_at(lambda t:t.func.signal_diffusion.diffusion_constants.weight,filter_spec,replace=True)
		return filter_spec
	
	def label_not_diffusive(tree):
		# Returns True for the parameters that are NOT the non-negative diffusive terms
		filter_spec = jax.tree_util.tree_map(lambda _:True,tree)
		filter_spec = eqx.tree_at(lambda t:t.func.signal_diffusion.diffusion_constants.weight,filter_spec,replace=False)
		return filter_spec
	
	opt_ra = optax.masked(opt_ra,label_not_diffusive)
	opt_d = optax.masked(opt_d,label_diffusive)
	
	
	return optax.chain(opt_d,opt_ra)
import jax
import equinox as eqx
import einops
from jaxtyping import Array, Float
from Common.model.spatial_operators import Ops
class Cell_motility(eqx.Module):
    motility_layers: list
    chemotaxis_layers:list
    ops: eqx.Module
    CELL_CHANNELS: int
    SIGNAL_CHANNELS: int
    TOTAL_CHANNELS: int
    
    


    def __init__(self,CELL_CHANNELS,SIGNAL_CHANNELS,PADDING,dx,INTERNAL_ACTIVATION,OUTER_ACTIVATION,INIT_SCALE,key):
        self.CELL_CHANNELS = CELL_CHANNELS
        self.SIGNAL_CHANNELS = SIGNAL_CHANNELS
        self.TOTAL_CHANNELS = CELL_CHANNELS + SIGNAL_CHANNELS
        
        key1,key2,key3,key4 = jax.random.split(key,4)
        self.ops = Ops(PADDING=PADDING,dx=dx)

        
        self.motility_layers = [
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.TOTAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key1
            ),
            INTERNAL_ACTIVATION,
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.CELL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key2
            ),
            OUTER_ACTIVATION
        ]
        
        self.chemotaxis_layers = [
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.TOTAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key3
            ),
            INTERNAL_ACTIVATION,
            eqx.nn.Conv2d(
                in_channels=self.TOTAL_CHANNELS,
                out_channels=self.CELL_CHANNELS*self.SIGNAL_CHANNELS,
                kernel_size=1,
                padding=0,
                use_bias=False,
                key=key4
            ),
            OUTER_ACTIVATION
        ]
        where = lambda l:l.weight
        
        self.chemotaxis_layers[0] = eqx.tree_at(where,
                                                 self.chemotaxis_layers[0],
                                                 INIT_SCALE*jax.random.normal(key=key1,shape=self.chemotaxis_layers[0].weight.shape))
        self.motility_layers[0] = eqx.tree_at(where,
                                                 self.motility_layers[0],
                                                 INIT_SCALE*jax.random.normal(key=key2,shape=self.motility_layers[0].weight.shape))

        self.chemotaxis_layers[-2] = eqx.tree_at(where,
                                                 self.chemotaxis_layers[-2],
                                                 INIT_SCALE*jax.random.normal(key=key3,shape=self.chemotaxis_layers[-2].weight.shape))
        self.motility_layers[-2] = eqx.tree_at(where,
                                                 self.motility_layers[-2],
                                                 INIT_SCALE*jax.random.normal(key=key4,shape=self.motility_layers[-2].weight.shape))

    def motility(self,X:Float[Array, "C x y"])->Float[Array, "C x y"]:
        for L in self.motility_layers:
            X = L(X)
        #X = einops.repeat(X,"cell x y -> dim cell x y",dim=2)
        return X
    
    def chemotaxis(self,X:Float[Array, "C x y"])->Float[Array, "2 cell signal x y"]:
        for L in self.chemotaxis_layers:
            X = L(X)
        X = einops.rearrange(X,"(cell signal) x y -> cell signal x y",cell=self.CELL_CHANNELS,signal=self.SIGNAL_CHANNELS)
        return X

    def __call__(self,X: Float[Array, "C x y"])->Float[Array, "cell x y"]:
        
        # Motility (anisotropic diffusion)
        cells = X[:self.CELL_CHANNELS]
        k_motility = self.motility(X)
       
        mot = self.ops.NonlinearDiffusion(k_motility,cells)

        # Chemotaxis - nasty bit!
        #(grad cells dot k_checm grad signals) + cells * (grad k_chem dot grad signals + k_chem laplacian signals)

        signals=X[self.CELL_CHANNELS:]
        k_chemotaxis= self.chemotaxis(X)

        A = einops.einsum(self.ops.Grad(cells),
                          k_chemotaxis,
                          self.ops.Grad(signals),
                          "dim cell x y, cell signal x y, dim signal x y -> cell x y")
        flat_k_chem = einops.rearrange(k_chemotaxis,
                                       "cell signal x y -> (cell signal) x y",
                                       cell=self.CELL_CHANNELS,
                                       signal=self.SIGNAL_CHANNELS)
        
        grad_k_chem = einops.rearrange(self.ops.Grad(flat_k_chem),
                                       "dim (cell signal) x y -> dim cell signal x y",
                                       cell=self.CELL_CHANNELS,
                                       signal=self.SIGNAL_CHANNELS)
        B = einops.einsum(grad_k_chem,
                          self.ops.Grad(signals),
                          "dim cell signal x y, dim signal x y -> cell x y")
        C = einops.einsum(k_chemotaxis,
                          self.ops.Lap(signals),
                          "cell signal x y, signal x y -> cell x y")

        return mot - (A + cells*(B+C))
        

    def partition(self):
        # total_diff,total_static = eqx.partition(self,eqx.is_array)
        # gc_diff,gc_static = self.grad_cells.partition()
        # gs_diff,gs_static = self.grad_signals.partition()
        # gm_diff,gm_static = self.grad_chemotaxis.partition()
        # ad_diff,ad_static = self.anisotropic_diffusion_cells.partition()
        # ls_diff,ls_static = self.laplacian_signals.partition()
        

        # where_grad_cell = lambda m: m.grad_cells
        # where_grad_signals = lambda m: m.grad_signals
        # where_grad_mix = lambda m: m.grad_chemotaxis
        # where_an_diff_cells = lambda m: m.anisotropic_diffusion_cells
        # where_ls_cells = lambda m: m.laplacian_signals
        

        # total_diff = eqx.tree_at(where_grad_cell,total_diff,gc_diff)
        # total_diff = eqx.tree_at(where_grad_signals,total_diff,gs_diff)
        # total_diff = eqx.tree_at(where_grad_mix,total_diff,gm_diff)
        # total_diff = eqx.tree_at(where_an_diff_cells,total_diff,ad_diff)
        # total_diff = eqx.tree_at(where_ls_cells,total_diff,ls_diff)
        
        # total_static = eqx.tree_at(where_grad_cell,total_static,gc_static)
        # total_static = eqx.tree_at(where_grad_signals,total_static,gs_static)
        # total_static = eqx.tree_at(where_grad_mix,total_static,gm_static)
        # total_static = eqx.tree_at(where_an_diff_cells,total_static,ad_static)
        # total_static = eqx.tree_at(where_ls_cells,total_static,ls_static)
        # return total_diff,total_static
    
        total_diff,total_static = eqx.partition(self,eqx.is_array)
        op_diff,op_static = self.ops.partition()
        where_ops = lambda m: m.ops
        total_diff = eqx.tree_at(where_ops,total_diff,op_diff)
        total_static = eqx.tree_at(where_ops,total_static,op_static)
        return total_diff,total_static
    
    def combine(self,diff,static):
        self = eqx.combine(diff,static)
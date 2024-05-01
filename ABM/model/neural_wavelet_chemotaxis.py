import jax
import jax.numpy as np
import equinox as eqx
from ABM.model.neural_chemotaxis import NeuralChemotaxis
from ABM.model.agent_nn_angle import agent_nn


class NeuralWaveletChemotaxis(NeuralChemotaxis):
    LENGTH_SCALE: float
    def __init__(self,LENGTH_SCALE,*args,**kwargs):
        NeuralChemotaxis.__init__(self,*args,**kwargs)
        self.Agent_nn = agent_nn(self.N_CHANNELS)
        self.LENGTH_SCALE = LENGTH_SCALE


    @eqx.filter_jit
    def _sense_pheremones(self,agents,pheremone_lattice):
        """
        Detect pheremones and gradients of pheremones at agent location

        Parameters
        ----------
        agents : (array[2,n_agents],array[2,n_agents])
            Tuple containing positions and velocities of all agents.
        pheremone_lattice : array[channels,grid_size,grid_size]
            Lattice for storing concentration of signalling pheremones.
        sensor_angle : float, optional
            Angle between different sensor regions. The default is 0.6.
        sensor_length : float, optional
            Distance between agent and sensor regions. The default is 3.

        Returns
        -------
        pheremone_weights: (array[channels,n_agents],array[channels,n_agents],array[channels,n_agents])
            Sum of pheremone concentrations in each sensor region for each agent

        """
        
        
        def sense_zone_ind(pos,vel):
            """
            Sense pheremones and gradients of them at a given agent position and direction

            VMAP THIS OVER AGENTS

            Parameters
            ----------
            pos : array[2]
                    positions
            vel : array[2]
                velocities
            

            Returns
            -------
            weights: array[3*channels + 2]
                array storing local pheremone averages, gradients of pheremones and orientation
            

            """

            theta = np.arctan2(vel[1],vel[0])

            sigma_x = int(self.LENGTH_SCALE)
            sigma_y = int(self.LENGTH_SCALE) 
            nstds = 2
            Lambda = 10000
            
            (y, x) = np.meshgrid(np.arange(-nstds*sigma_y, nstds*sigma_y + 1), np.arange(-nstds*sigma_x, nstds*sigma_x + 1))
            # Rotation
            x_theta = x * np.cos(theta) + y * np.sin(theta)
            y_theta = -x * np.sin(theta) + y * np.cos(theta)

            gb_x = np.exp(
                -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
            ) * np.sin(2 * np.pi / Lambda * x_theta) 
            gb_y = np.exp(
                -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
            ) * np.sin(2 * np.pi / Lambda * y_theta) 
            p_x = gb_x / np.sum(np.abs(gb_x))
            p_y = gb_y / np.sum(np.abs(gb_y))




            #p_x,p_y = self._gabor_wavelets(c_angle)
            filter_width = p_x.shape[0]//2
            c = np.rint(pos).astype(int)
            #x_ind = np.arange(c[0]-filter_width,c[0]+filter_width)
            #y_ind = np.arange(c[1]-filter_width,c[1]+filter_width)
            x_ind = np.stack((c[0]-4,c[0]-3,c[0]-2,c[0]-1,c[0],c[0]+1,c[0]+2,c[0]+3,c[0]+4))#%self.GRID_SIZE
            y_ind = np.stack((c[1]-4,c[1]-3,c[1]-2,c[1]-1,c[1],c[1]+1,c[1]+2,c[1]+3,c[1]+4))#%self.GRID_SIZE
            
            if self.PERIODIC:
                x_ind = x_ind%self.GRID_SIZE
                y_ind = y_ind%self.GRID_SIZE
                
                _xs,_ys = np.meshgrid(x_ind,y_ind)
                id = np.sum(pheremone_lattice[:,_xs,_ys],axis=(1,2)) # [:] over channels
                w_x = np.sum(pheremone_lattice[:,_xs,_ys]*p_x,axis=(1,2)) # [:] over channels
                w_y = np.sum(pheremone_lattice[:,_xs,_ys]*p_y,axis=(1,2)) # [:] over channels
            else:
                x_ind = x_ind + filter_width
                y_ind = y_ind + filter_width
                _xs,_ys = np.meshgrid(x_ind,y_ind)
                padded_pheremone_lattice = np.pad(pheremone_lattice,((0,0),(filter_width,filter_width),(filter_width,filter_width)),constant_values=0.0)
                id = np.sum(padded_pheremone_lattice[:,_xs,_ys],axis=(1,2)) # [:] over channels
                w_x = np.sum(pheremone_lattice[:,_xs,_ys]*p_x,axis=(1,2)) # [:] over channels
                w_y = np.sum(pheremone_lattice[:,_xs,_ys]*p_y,axis=(1,2)) # [:] over channels
            
            
            cos_angle = np.cos(theta)
            sin_angle = np.sin(theta)
            return id,w_x,w_y,cos_angle[np.newaxis],sin_angle[np.newaxis]
        v_sense = jax.vmap(sense_zone_ind,(0,0),(0,0,0,0,0),axis_name="N_AGENTS")
        #print(agents.shape)
        weights =v_sense(agents[0].T,agents[1].T)
        
        return weights


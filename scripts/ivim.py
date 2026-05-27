import numpy as np
from scripts.network import VesselNetwork
from scripts.trajectories_utils import get_particle_path
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from scipy.optimize import curve_fit
from scripts.visualize import plot_ivim_tensors

class IVIMSimulator:
    """Simulates IVIM MRI signal based on particle trajectories through a vascular network and diffusion in the extravascular space."""
    def __init__(self, vessel_net: VesselNetwork, diffusion_time=100e-3, delta_t=1e-4, 
                 grad_dur=25e-3, pixel_size=1e-6, D_input=1000):
        """
        Parameters:
            vessel_net (VesselNetwork): An instance of the VesselNetwork class containing the vascular graph and flow information
            diffusion_time (float): diffusion time in seconds (default: 100e-3 s)
            delta_t (float): time step for particle simulation in seconds (default: 1e-4 s)
            grad_dur (float): duration of the one diffusion gradient lobe in seconds (default: 25e-3 s)
            pixel_size (float): size of one pixel in meters (default: 1e-6 m)
            D_input (float or array_like): Diffusion coefficient for the extravascular space. Can be a single scalar for isotropic diffusion, or a 3x3 array for an anisotropic diffusion tensor (default: 1000 mm^2/s)
        """
        # Check if vessel net has flows
        if not vessel_net.data.get('flow_calculated', False):
            raise ValueError("Vessel network must have flow calculated to simulate perfusion.")
        self.vessel_net = vessel_net
        self.diffusion_time = diffusion_time
        self.delta_t = delta_t
        self.grad_dur = grad_dur
        self.pixel_size = pixel_size
        self.D_input = D_input

        self.gamma = 42.58e6 * 2 * np.pi
        self.simulation_time = self.diffusion_time + self.grad_dur
        
        self.perfusion_trajectories = []
        self.diffusion_trajectories = []
        self.ivim_signal = None
        self.b_values = None
        self.gradient_directions = None
        self.b_threshold = None
        self.f_tensor = None
        self.D_tensor = None
        self.D_star_tensor = None


    #================================================================================
    # Trajectory generation =========================================================
    #================================================================================
    def generate_n_diffusing_particles(self, n):
        """ Generates trajectories for n diffusing particles."""
        if np.isscalar(self.D_input): # isotropic
            D_tensor = np.eye(3) * self.D_input
        else: # anisotropic
            D_tensor = np.array(self.D_input)
        cov = 2 * D_tensor * self.delta_t
        n_steps = int(self.simulation_time / self.delta_t)
        # Generate all steps at once for efficiency using numpy's multivariate normal generator
        steps = np.random.multivariate_normal(
            mean=[0, 0, 0], 
            cov=cov, 
            size=(n, n_steps)
        )
        start_pos = np.zeros((n, 1, 3))
        self.diffusion_trajectories = np.concatenate((start_pos, np.cumsum(steps, axis=1)), axis=1)
        return self.diffusion_trajectories
    
    def generate_n_perfusing_particles(self, n):
        """ Generates trajectories for n perfusing particles.
        
        Parameters:
            n (int): The number of perfusing particles to simulate.

        Returns:
            trajectories (np.ndarray): An array of shape (n, n_steps, 3) containing the trajectories of the perfusing particles in pixel coordinates.
        """
        self.perfusion_trajectories = []
        min_pxs_vel = 20 # 20px/s = 0.02mm/s, leave out extremely slow trajectories, these artifacts can appear when calculating velocities
        while len(self.perfusion_trajectories) < n:
            path, opustila_graf = get_particle_path(self.vessel_net.data, time=self.simulation_time, delta_time=self.delta_t)
            if path is not None and len(path) > 1 and not opustila_graf: # Only trajectories that stayed in the graph
                displacements = [np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1)]
                avg_velocity = np.mean(displacements) / self.delta_t if len(displacements) > 0 else 0
                if avg_velocity > min_pxs_vel: 
                    self.perfusion_trajectories.append(path)
                    if len(self.perfusion_trajectories) % 500 == 0: print(f"Generated {len(self.perfusion_trajectories)}/{n}")
        self.perfusion_trajectories = np.array(self.perfusion_trajectories)
        self.perfusion_trajectories = self.perfusion_trajectories[:, :, [2, 1, 0]] # Switch x and z axis - so that the axis order corresponds to original tiff data
        return self.perfusion_trajectories
    
    #================================================================================
    # IVIM signal calculation =========================================================
    #================================================================================
    
    def _get_G_amplitude(self, b):
        """Calculate the required gradient amplitude for a given b-value based on the Stejskal-Tanner equation."""
        # Amplituda pro dané b
        if b == 0:
            G_amplitude = 0
        else:
            b_si = b * 1e6 # Convert from mm^2/s to m^2/s
            Delta = self.simulation_time - self.grad_dur 
            denom = (self.gamma**2) * (self.grad_dur**2) * (Delta - self.grad_dur/3.0)
            G_amplitude = np.sqrt(b_si / denom)
        return G_amplitude

    def get_signal_one_direction(self, gradient_direction, all_trajectories):
        """Calculate the IVIM signal for a single gradient direction given the particle trajectories."""
        # Combine both trajectory types into a single array
        traj_array = all_trajectories
        traj_array = traj_array.astype(np.float32) # float32 for faster memory operations
        _, n_steps, _ = traj_array.shape
        
        # Convert coordinates from pixels to meters (required for the physics formula)
        pos_m = traj_array * self.pixel_size 
        
        # Normalize the direction vector
        g_dir = np.array(gradient_direction, dtype=np.float32)
        g_dir /= np.linalg.norm(g_dir)
        
        # 3. GRADIENT SHAPE G(t): Stejskal-Tanner sequence
        # Create the bipolar gradient profile mask. First lobe is positive, second is negative.
        time_points = np.linspace(0, self.simulation_time, num=n_steps)
        g_mask = np.zeros((n_steps, 3), dtype=np.float32)
        g_mask[(time_points >= 0) & (time_points <= self.grad_dur)] = g_dir
        g_mask[(time_points >= self.simulation_time - self.grad_dur) & (time_points <= self.simulation_time)] = -g_dir
        
        # Sum of dot products over time
        # gamma * Delta_t * sum(p(t) . G_mask(t))
        # np.einsum rapidly computes the dot product of position and gradient for every 
        # particle at every time step, and sums it over the time axis (t) simultaneously.
        # (N_particles, N_steps, 3) x (N_steps, 3) -> (N_particles,)
        phi_unit = self.gamma * np.einsum('ptc,tc->p', pos_m, g_mask) * self.delta_t
        mean_signals = []
        
        # Compute signal for all b-values
        for b in self.b_values:
            g_amp = self._get_G_amplitude(b)
            
            # phi_unit was calculated with a unit gradient (amplitude 1).
            # Here it is scaled by the actual amplitude G_amp for the given 'b'.
            # Final accumulated phase: phi = gamma * Delta_t * sum(p(t) . G_amp(t))
            phi = phi_unit * g_amp
            # S = mean(exp(-i * phi))
            # Calculate the complex signal for each particle and average them (macroscopic voxel)
            signal = np.abs(np.mean(np.exp(-1j * phi)))
            mean_signals.append(signal)
        return mean_signals

    def get_ivim_signal(self, n_perfusion, n_diffusion, b_values, gradient_directions, out_file = None):
        """
        Generates particle trajectories and calculates the full IVIM MRI signal.

        Parameters:
            n_perfusion (int): Number of perfusing particles to simulate within the vascular network.
            n_diffusion (int): Number of diffusing particles to simulate in the extravascular space.
            b_values (array_like): A list or 1D numpy array of b-values (in s/mm^2) for the sequence.
            gradient_directions (array_like): A 1D array (for a single direction) or 2D array (for multiple directions) representing the gradient vectors [x, y, z]. These are automatically normalized.
            out_file (str, optional): Path to a .npz file where the computed signals, parameters, and setup details will be saved. Default is None (no saving).

        Returns:
            signals (np.ndarray): A list containing the computed signal arrays. Each element in the list 
            corresponds to one gradient direction and contains the signal values 
            matching the sequence of requested b-values.
        """
        self.b_values = b_values
        self.gradient_directions = gradient_directions

        print("Generating particle trajectories - may take a few minutes...")
        self.generate_n_perfusing_particles(n_perfusion)
        self.generate_n_diffusing_particles(n_diffusion)
        
        if len(self.perfusion_trajectories) > 0 and len(self.diffusion_trajectories) > 0:
            all_trajectories = np.concatenate((self.perfusion_trajectories, self.diffusion_trajectories), axis=0)
        elif len(self.perfusion_trajectories) > 0:
            all_trajectories = self.perfusion_trajectories
        else:
            all_trajectories = self.diffusion_trajectories

        self.gradient_directions = np.array(self.gradient_directions)
        if len(self.gradient_directions.shape) == 1:
            self.gradient_directions = self.gradient_directions.reshape(1, -1)
        self.gradient_directions = self.gradient_directions / np.linalg.norm(self.gradient_directions, axis=1, keepdims=True)
        mean_signals = []
        for grad_dir in self.gradient_directions:
            signals_for_dir = self.get_signal_one_direction(grad_dir, all_trajectories)
            mean_signals.append(signals_for_dir)
        all_signals_array = np.array(mean_signals)
        if out_file is not None:
            print(f"Saving signals to {out_file}...")
            np.savez_compressed(
            out_file, 
            signals=all_signals_array, 
            b_values=self.b_values, 
            directions=self.gradient_directions,
            params={'simulation_time': self.simulation_time, 'diffusion_time': self.diffusion_time, 'n_particles_perfusion': n_perfusion, 'n_particles_diffusion': n_diffusion, 'D_input': self.D_input, 'delta_t': self.delta_t, 'grad_dur': self.grad_dur}
            )

        self.ivim_signal = all_signals_array
        return all_signals_array
    
    #================================================================================
    # IVIM tensor extraction =========================================================
    #================================================================================
    def _custom_segmented_ivim_fit(self, single_dir_signal):
        """Performs a segmented fit of the IVIM model to the signal for a single gradient direction, extracting f, D, and D* parameters."""
        # Signal normalization
        signal_norm = single_dir_signal / single_dir_signal[0]

        # Diffusion fit (D and f)
        high_mask = self.b_values >= self.b_threshold
        b_high = self.b_values[high_mask]
        s_high = signal_norm[high_mask]

        # S = (1-f)*exp(-b*D) -> ln(S) = ln(1-f) - b*D
        p = np.polyfit(b_high, np.log(s_high), 1)
        D_final = np.clip(-p[0], 0.0, 0.005)
        f_final = np.clip(1.0 - np.exp(p[1]), 0.001, 0.99)
        
        # Perfusion fit
        low_mask = self.b_values <= self.b_threshold
        b_low = self.b_values[low_mask]
        
        # Deduction of diffusion component from low b-values
        s_diff = (1.0 - f_final) * np.exp(-b_low * D_final)
        s_perf_low = signal_norm[low_mask] - s_diff

        def perf_only(b, D_star):
            return f_final * np.exp(-b * D_star)
        popt, _ = curve_fit(perf_only, b_low, s_perf_low, p0=[0.01], bounds=(0.001, 0.5))
        D_star_final = popt[0]
        return f_final, D_final, D_star_final


    def _get_parameter_arrays_scipy(self):
        """Fits the IVIM model to the signal for all gradient directions and returns arrays of f, D, and D* parameters."""
        f_list = []
        D_list = []
        D_star_list = []

        # Fitting f, D and D* parameters for each gradient direction
        for i, direction in enumerate(self.gradient_directions):
            signal_1d = self.ivim_signal[i, :]
            fitted_f, fitted_D, fitted_D_star = self._custom_segmented_ivim_fit(signal_1d)
            f_list.append(fitted_f)
            D_list.append(fitted_D)
            D_star_list.append(fitted_D_star)  
        f_array = np.array(f_list)
        D_array = np.array(D_list)
        D_star_array = np.array(D_star_list)
        return f_array, D_array, D_star_array

    
    def _get_tensor_from_param_arrays(self, X, fake_b):
        """Uses DTI to fit a tensor to parrameter array (param results from all directions)."""
        fake_signal_b0 = np.array([1.0])
        b0_dir = np.array([[0.0, 0.0, 0.0]])
        fake_signal_b = np.exp(-X * fake_b)
        signals = np.concatenate([fake_signal_b0, fake_signal_b])
        bvals = np.concatenate([[0], np.repeat(fake_b, len(X))])
        bvecs = np.vstack([b0_dir, self.gradient_directions])
        gtab = gradient_table(bvals=bvals, bvecs=bvecs, b0_threshold=0)
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(signals)
        tensor_matrix = tenfit.quadratic_form 
        return tensor_matrix

    def _get_fa_md(self, tensor_matrix):
        """Calculate fractional anisotropy (FA) and mean eigenvalue from a tensor."""
        eigenvalues, eigenvectors = np.linalg.eigh(tensor_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        l1, l2, l3 = eigenvalues
        fa_numerator = np.sqrt((l1 - l2)**2 + (l2 - l3)**2 + (l3 - l1)**2)
        fa_denominator = np.sqrt(l1**2 + l2**2 + l3**2)
        if fa_denominator > 0:
            fa = np.sqrt(1/2) * (fa_numerator / fa_denominator)
        else:
            fa = 0.0
        md = (l1 + l2 + l3) / 3
        return fa, md

    def fit_ivim_and_get_tensors(self, out_path=None, b_threshold=175):
        """Fits the IVIM model to the simulated signal and extracts diffusion tensors for f, D, and D* parameters. Optionally saves the fitted tensors and their FA/MD values to a file."""
        if self.ivim_signal is None:
            raise ValueError("IVIM signal not calculated yet. Please run get_ivim_signal() first.")
        self.b_threshold = b_threshold
        f_array, D_array, D_star_array = self._get_parameter_arrays_scipy()
        # Calculate fake b values for tensor fitting based on the average parameter values, to ensure the signal is in a reasonable range for DTI fitting
        avg_f = np.mean(f_array)
        fake_b_f = 1.0 / avg_f
        f_tensor = self._get_tensor_from_param_arrays(f_array, fake_b=fake_b_f)
        avg_d_star = np.mean(D_star_array)
        fake_b_d_star = 1.0 / avg_d_star
        D_star_tensor = self._get_tensor_from_param_arrays(D_star_array, fake_b = fake_b_d_star)
        avg_d = np.mean(D_array)
        fake_b_d = 1.0 / avg_d
        D_tensor = self._get_tensor_from_param_arrays(D_array, fake_b = fake_b_d)
        if out_path is not None:
            fa_f, md_f = self._get_fa_md(f_tensor)
            fa_D_star, md_D_star = self._get_fa_md(D_star_tensor)
            fa_D, md_D = self._get_fa_md(D_tensor)
            np.savez(out_path, f_tensor=f_tensor, D_star_tensor=D_star_tensor, D_tensor=D_tensor, fa_f=fa_f, md_f=md_f, fa_D_star=fa_D_star, md_D_star=md_D_star, fa_D=fa_D, md_D=md_D)
            print(f"Saved fitted tensors to {out_path}")
        self.f_tensor = f_tensor
        self.D_star_tensor = D_star_tensor
        self.D_tensor = D_tensor
        return f_tensor, D_star_tensor, D_tensor
    
    def show_tensors(self):
        """Visualizes the fitted IVIM tensors in the context of the vascular network."""
        if self.f_tensor is None or self.D_tensor is None or self.D_star_tensor is None:
            raise ValueError("Tensors not fitted yet. Please run fit_ivim_and_get_tensors() first.")
        plot_ivim_tensors(self.vessel_net.data, self.f_tensor, self.D_star_tensor, self.D_tensor)

    def print_evaluation(self):
        """Prints evaluation metrics for the fitted f tensor, including its FA and MD values, and the deviation of its principal direction from the reference direction in the vessel network."""
        if self.f_tensor is None or self.D_tensor is None or self.D_star_tensor is None:
            raise ValueError("Tensors not fitted yet. Please run fit_ivim_and_get_tensors() first.")
        if self.vessel_net.data['fa'] is None:
            raise ValueError("Vessel network does not have FA calculated. Please run orientational_analysis() on the vessel network first.")
        fa_f, md_f = self._get_fa_md(self.f_tensor)
        print("=" * 40)
        print("F TENSOR EVALUATION")
        print("=" * 40)
        print("Fractional anisotropy of f tensor:", fa_f)
        print("Reference fractional anisotropy:", self.vessel_net.data['fa'])
        print("Mean diffusivity of f tensor:", md_f)
        evals, evecs = np.linalg.eigh(self.f_tensor)
        principal_vec = evecs[:, -1]
        ref_vec = self.vessel_net.data['principal_direction']
        dot_product = np.abs(np.dot(ref_vec, principal_vec))
        deviation = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        print(f"Deviation of principal direction of f tensor from reference: {deviation:.2f} degrees")


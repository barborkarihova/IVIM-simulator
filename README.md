# IVIM signal simulator from vessel network

This code is used to simulate the IVIM (Intravoxel Incoherent Motion) signal in vascular networks.

The project was developed using Python 3.12. All required dependencies are stated in `requirements.txt`.


## The project is focused on:

* **Hydrodynamics:** Calculation of blood flow and velocity from a vascular network model
* **Trajectories:** Generation of perfusing (intravascular) and diffusing (extravascular) particle trajectories
* **Signal Simulation:** IVIM MRI signal generation based on simulated trajectories
* **Tensor Extraction:** Fitting of IVIM parameters (f, D, D*) and construction of the resulting IVIM tensors



## Project structure:

* 💻 **`usage_example.ipynb` - Code demonstration** 💻
* **`networks/`** - Contains `example_network.json`. *(As datasets used in thesis were internal, the example network is from an online source [here](https://zenodo.org/records/269650)).*
* **`scripts/`**
  * `network.py` - Network preprocessing and flow calculations
  * `ivim.py` - Particle tracking, IVIM simulation and tensor fitting
  * `trajectories_utils.py` - Helper functions for particle trajectory tracking
  * `visualize.py` - Visualization of paths, signals, and fitted tensors
import numpy as np 
from numpy.typing import ArrayLike
import h5py 
import time 

from pathlib import Path
from scipy import special
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline

"""
Constrain log10Mmin from ng
Used when generating HOD parameter sets. 
For each parameter set, we estimate log10Mmin that yields ng = ng_desired.
"""


D13_BASE_PATH = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files"
BOX_VOLUME = 2000.0**3


def compute_ng_analytical(log10Mmin, sigma_logM, log10M1, kappa, alpha, mass_center, dn_dM):
    """
    Computes the analytical ng from HOD parameters and HMF.
    log10Mmin can either be scalar or a 1D array. 
    """
    ### Compute ng analytically from HOD params and HMF
    mass_center         = mass_center[:,np.newaxis]
    Nc                  = 0.5 * special.erfc(np.log10(10**log10Mmin / mass_center) / sigma_logM)
    mask                = mass_center > kappa * 10**log10Mmin # No satellite galaxies below Mmin

    lambda_values = ((mass_center - kappa * 10**log10Mmin) / 10**log10M1)**alpha 
    lambda_values[mask==False] = 0.0
    Ns = Nc * lambda_values

    return simpson(dn_dM[:,np.newaxis] * (Nc + Ns), mass_center, axis=0)



def estimate_log10Mmin_from_gal_num_density(
        sigma_logM_array: ArrayLike,
        log10M1_array:    ArrayLike,
        kappa_array:      ArrayLike,
        alpha_array:      ArrayLike,
        version:          int,
        phase:            int,
        ng_desired:       float = 2.174e-4, 
        ) -> ArrayLike:
    
    """
    Finds the appropriate value of log10Mmin that yields ng = ng_desired.
    Computes ng with Eq. (19) in https://doi.org/10.1093/mnras/stad1207
    """

    # Halo data 
    version_dir      = f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}"
    SIMULATION_DIR   = f"{D13_BASE_PATH}/{version_dir}"
    POS_VEL_MASS_ARRAYS_PATH = f"{SIMULATION_DIR}/pos_vel_mass_arrays"
    if not Path(POS_VEL_MASS_ARRAYS_PATH).exists():
        print(f"Error: {POS_VEL_MASS_ARRAYS_PATH} does not exist")
        print(" - Halo pos,vel,mass must be stored in this directory to constrain ng.")
        return


    # Load halo mass array
    mass = np.load(f"{POS_VEL_MASS_ARRAYS_PATH}/L1_mass.npy") # shape: (N_halos,)
    
    
    # Compute HMF from halo catalogue 
    N           = 201
    mass_edges  = np.logspace(np.log10(np.min(mass)), np.log10(np.max(mass)), N)
    num_halos   = np.histogram(mass, mass_edges)[0]
    mass_center = 0.5 * (mass_edges[1:] + mass_edges[:-1])
    dn_dM       = num_halos /  BOX_VOLUME / (mass_edges[1:] - mass_edges[:-1])

    # Estimate ng from halo catalogue
    # The log10Mmin range chosen has been tested. 
    # it yields ng values around 2.174e-4 for all parameter sets.
    N_Mmin        = 300
    log10Mmin_arr = np.linspace(13.0, 14.5, N_Mmin) 

    # Number of parameter sets
    N_params            = len(sigma_logM_array)
    log10Mmin_best_fit  = np.zeros(N_params)

    # Loop through parameter sets and estimate log10Mmin that yields ng = ng_desired
    for i in range(N_params):
        # Compute ng for all values of log10Mmin
        ng_arr          = compute_ng_analytical(log10Mmin_arr, 
                                                sigma_logM_array[i], 
                                                log10M1_array[i], 
                                                kappa_array[i], 
                                                alpha_array[i], 
                                                mass_center, 
                                                dn_dM)
        
        # Use spline interpolation to find log10Mmin for ng = ng_desired
        # log10Mmin is a very well-behaved function 
        ng_sorted_idx    = np.argsort(ng_arr)
        log10Mmin_spline = UnivariateSpline(
            ng_arr[ng_sorted_idx], 
            log10Mmin_arr[ng_sorted_idx], 
            k=3, 
            s=0
            )
        log10Mmin_best_fit[i] = log10Mmin_spline(ng_desired)

    return log10Mmin_best_fit


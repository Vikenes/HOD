import numpy as np 
from numpy.typing import ArrayLike
import h5py 
import time 
from scipy.optimize import fsolve

from pathlib import Path
from scipy import special
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline

"""
Constrain log10Mmin from ng
Used when generating HOD parameter sets. 
For each parameter set, we estimate log10Mmin that yields ng = ng_desired.
"""

# Suppress runtime warning from invalid value encountered in power 
np.seterr(invalid='ignore')

D13_BASE_PATH    = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files"
BOX_VOLUME       = 2000.0**3
BOX_VOLUME_SMALL = 500.0**3



def get_log10Mmin_from_fixed_ng(
        sigma_logM_array: ArrayLike,
        log10M1_array:    ArrayLike,
        kappa_array:      ArrayLike,
        alpha_array:      ArrayLike,
        version:          int,
        phase:            int,
        ng_desired:       float, 
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
        # log10Mmin_best_fit[i] = log10Mmin_spline(ng_desired)
        log10Mmin_best_fit[i] = log10Mmin_spline(ng_desired)


    return log10Mmin_best_fit


def compute_ng_analytical(log10Mmin, sigma_logM, log10M1, kappa, alpha, mass_center, dn_dM):
    """
    Computes the analytical ng from HOD parameters and HMF.
    log10Mmin can either be scalar or a 1D array. 
    """
    ### Compute ng analytically from HOD params and HMF
    mass_center         = mass_center[:,np.newaxis]
    Nc                  = 0.5 * special.erfc(np.log10(10**log10Mmin / mass_center) / sigma_logM)
    mask                = mass_center > kappa * 10**log10Mmin # No satellite galaxies below Mmin

    # lambda_values = ((mass_center - kappa * 10**log10Mmin) / 10**log10M1)**alpha 
    # lambda_values[mask==False] = 0.0
    lambda_values = np.where(mask, ((mass_center - kappa * 10**log10Mmin) / 10**log10M1)**alpha, 0.0)

    Ns = Nc * lambda_values

    return simpson(dn_dM[:,np.newaxis] * (Nc + Ns), mass_center, axis=0)



def ng_func_array(log10Mmin, sigma_logM, log10M1, kappa, alpha, ng_array, mass_center, dn_dM):
    """
    Callable func for scipy fsolve to find log10Mmin that yields ng = ng_desired.
    """
    ### Compute ng analytically from HOD params and HMF
    mass_center         = mass_center[:,np.newaxis]
    # mass_center = np.hstack([mass_center[:,np.newaxis]] * len(log10Mmin))
    Nc                  = 0.5 * special.erfc(np.log10(10**log10Mmin / mass_center) / sigma_logM)
    mask                = mass_center > kappa * 10**log10Mmin # No satellite galaxies below Mmin

    # lambda_values = np.zeros_like(mask, dtype=float)
    lambda_values = np.where(mask, ((mass_center - kappa * 10**log10Mmin) / 10**log10M1)**alpha, 0.0)
  
    Ns = Nc * lambda_values
  

    return simpson(dn_dM[:,np.newaxis] * (Nc + Ns), mass_center, axis=0) - ng_array


def ng_func_scalar(log10Mmin, sigma_logM, log10M1, kappa, alpha, ng_desired, mass_center, dn_dM):
    """
    Callable func for scipy fsolve to find log10Mmin that yields ng = ng_desired.
    """
    ### Compute ng analytically from HOD params and HMF
    Nc                  = 0.5 * special.erfc(np.log10(10**log10Mmin / mass_center) / sigma_logM)
    mask                = mass_center > kappa * 10**log10Mmin # No satellite galaxies below Mmin

    # lambda_values = np.zeros_like(mask, dtype=float)
    lambda_values = np.where(mask, ((mass_center - kappa * 10**log10Mmin) / 10**log10M1)**alpha, 0.0)
  
    Ns = Nc * lambda_values
  

    return simpson(dn_dM * (Nc + Ns), mass_center, axis=0) - ng_desired


def get_log10Mmin_from_varying_log_ng(
        sigma_logM_array: ArrayLike,
        log10M1_array:    ArrayLike,
        kappa_array:      ArrayLike,
        alpha_array:      ArrayLike,
        log_ng_array:     ArrayLike,
        version:          int,
        phase:            int,
        ) -> ArrayLike:
    
    """
    Finds the appropriate values of log10Mmin that yield ng = 10**log_ng_array.
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

    logMmin_init = np.ones_like(sigma_logM_array) * 13.0
    ng_array     = 10**log_ng_array

    # Estimate ng from halo catalogue
    root = fsolve(ng_func_array, logMmin_init, args=(sigma_logM_array, log10M1_array, kappa_array, alpha_array, ng_array, mass_center, dn_dM))

    return root 



def get_log10Mmin_from_fixed_log_ng_small(
        array_path: Path,
        params: tuple,
        ) -> ArrayLike:
    
    """
    Finds the appropriate values of log10Mmin that yield ng = 10**log_ng_array.
    Computes ng with Eq. (19) in https://doi.org/10.1093/mnras/stad1207
    """

    sigma_logM, log10M1, kappa, alpha, log_ng = params

    # Halo data 
    if not array_path.exists():
        raise FileNotFoundError(f"Error: {array_path} does not exist")

    # Load halo mass array
    mass = np.load(f"{array_path}/L1_mass.npy") # shape: (N_halos,)
    
    # Compute HMF from halo catalogue 
    N           = 201
    mass_edges  = np.logspace(np.log10(np.min(mass)), np.log10(np.max(mass)), N)
    num_halos   = np.histogram(mass, mass_edges)[0]
    mass_center = 0.5 * (mass_edges[1:] + mass_edges[:-1])
    dn_dM       = num_halos / BOX_VOLUME_SMALL / (mass_edges[1:] - mass_edges[:-1])

    logMmin_init = 13.0
    ng_desired   = 10**log_ng

    # Estimate ng from halo catalogue
    root = fsolve(ng_func_array, logMmin_init, args=(sigma_logM, log10M1, kappa, alpha, ng_desired, mass_center, dn_dM))

    return root[0] 
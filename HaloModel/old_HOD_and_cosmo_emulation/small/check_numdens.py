import h5py 
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd 
import time 
import concurrent.futures
import sys


sys.path.append("/uio/hume/student-u74/vetleav/Documents/thesis/HOD/HaloModel/HOD_and_cosmo_emulation")

BASE_PATH = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/small")

    
def print_logMmin():
    log10Mmin_array = np.load("log10Mmin_small_415.npy")
    mean = np.mean(log10Mmin_array)
    std  = np.std(log10Mmin_array)
    print(f"Mean: {mean:.4f} | Std: {std:.5f}")
    print(f"Min: {np.min(log10Mmin_array):.2f} | Max: {np.max(log10Mmin_array):.2f}")   


from numdens_HOD import estimate_log10Mmin_from_gal_num_density_small

def get_log10Mmin_small(phase, params):

    # HOD_parameters 
    sigma_logM, log10M1, kappa, alpha = params

    arrays_path = Path(BASE_PATH / f"AbacusSummit_small_c000_ph{phase}/pos_vel_mass_arrays")
    log10Mmin   = estimate_log10Mmin_from_gal_num_density_small(
        arrays_path         = arrays_path,
        sigma_logM_array    = sigma_logM,
        log10M1_array       = log10M1,
        kappa_array         = kappa,
        alpha_array         = alpha,
        )
    
    return log10Mmin

def save_N_number_densities(N):
    from numdens_HOD import estimate_log10Mmin_from_gal_num_density_small

    HOD_PARAMS_PATH  = f"/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files"
    HOD_PARAMS          = pd.read_csv(f"{HOD_PARAMS_PATH}/AbacusSummit_base_c000_ph000/HOD_parameters/HOD_parameters_fiducial_ng_fixed.csv")

    sigma_logM  = HOD_PARAMS["sigma_logM"][0]#0.6915 
    log10M1     = HOD_PARAMS["log10M1"][0]#14.42 # h^-1 Msun
    kappa       = HOD_PARAMS["kappa"][0]#0.51 
    alpha       = HOD_PARAMS["alpha"][0]#0.9168  
    params      = (sigma_logM, log10M1, kappa, alpha)

    # Generate N random integers in the half open interval [3000,5000) without duplicates
    np.random.seed(0)
    all_phases = np.arange(3000, 5000)
    phases = np.random.choice(all_phases, N, replace=False)
    # Loop over phases, remove values that are not directories
    for phase in phases:
        if not Path(BASE_PATH / f"AbacusSummit_small_c000_ph{phase}").is_dir():
            phases = np.delete(phases, np.where(phases == phase))
    N_phases = len(phases)
    logMmin_lst = [] #np.zeros_like(phases, dtype=float)
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(get_log10Mmin_small, phase, params) for phase in phases]
        for f in concurrent.futures.as_completed(results):
            # Add float value of f to list 
            logMmin_lst.append(f.result().item())

    dur = time.time() - t0 
    print(f"Time taken: {dur:.2f} s. Average over {N_phases} phases: {dur/N_phases:.2f} s")

    log10Mmin_array = np.array(logMmin_lst)
    np.save(f"log10Mmin_small_{N_phases}.npy", log10Mmin_array)

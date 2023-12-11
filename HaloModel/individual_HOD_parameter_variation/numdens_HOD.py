import numpy as np 
import h5py 
import time 
from dq import Cosmology
import hmd
from hmd.catalogue import ParticleCatalogue, HaloCatalogue, GalaxyCatalogue
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW
from hmd.populate import HODMaker
from pathlib import Path
from scipy import special
from scipy.integrate import simpson
import pandas as pd
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message='Astropy cosmology class contains massive neutrinos, which are not taken into account in Colossus.')

HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
HALO_ARRAYS_PATH = f"{HOD_DATA_PATH}/version0"

OUTFILEPATH = f"{HOD_DATA_PATH}/HOD_data"
dataset_names = ['train', 'test', 'val']

def compute_ng_analytical(log10Mmin, sigma_logM, log10M1, kappa, alpha, mass_center, dn_dM):
    ### Compute ng analytically from HOD params and HMF

    Nc = 0.5 * special.erfc(np.log10(10**log10Mmin / mass_center) / sigma_logM)
    mask = mass_center > kappa * 10**log10Mmin
    lambda_values = np.zeros_like(mass_center)
    lambda_values[mask] = ((mass_center[mask] - kappa * 10**log10Mmin) / 10**log10M1)**alpha

    Ns = Nc * lambda_values
    Ng = Nc + Ns

    integrand = dn_dM * Ng

    return simpson(integrand, mass_center)


def estimate_log10Mmin_from_gal_num_density(
        sigma_logM_array=None,
        log10M1_array=None,
        kappa_array=None,
        alpha_array=None,
        ng_desired=2.174e-4, 
        test=False,
        ):


    ### Make halo catalogue 
    # Define cosmology and simparams 
    cosmology = Cosmology.from_custom(run=0, emulator_data_path="/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/AbacusSummit_base_c130_ph000")
    redshift = 0.25 
    boxsize  = 2000.0
    # Halo data 
    pos  = np.load(f"{HALO_ARRAYS_PATH}/L1_pos.npy") 
    vel  = np.load(f"{HALO_ARRAYS_PATH}/L1_vel.npy")
    mass = np.load(f"{HALO_ARRAYS_PATH}/L1_mass.npy")
    # Make halo catalogue
    halocat = HaloCatalogue(
        pos,
        vel,
        mass,
        boxsize,
        conc_mass_model=hmd.concentration.diemer15,
        cosmology=cosmology,
        redshift=redshift,
        )
    
    # Compute HMF from halo catalogue 
    N = 200
    mmin = np.log10(np.min(mass))
    mmax = np.log10(np.max(mass))
    mass_edges = np.logspace(mmin, mmax, N)
    dn_dM = halocat.compute_hmf(mass_edges)
    mass_center = 0.5 * (mass_edges[1:] + mass_edges[:-1])

    # Estimate ng from halo catalogue
    N_Mmin = 200
    log10Mmin_arr = np.linspace(13.4, 13.9, N_Mmin) # Range yields ng values around 2.174e-4 for all parameter sets
    
    if test:
        ### FIDUCIAL VALUES. From https://arxiv.org/pdf/2208.05218.pdf, Table 3  ###
        # For testing purposes mainly. 
        sigma_logM_array = np.array([0.6915])
        log10M1_array    = np.array([14.42])
        kappa_array      = np.array([0.51])
        alpha_array      = np.array([0.9168])


    N_params = len(sigma_logM_array)
    log10Mmin_best_fit = np.zeros(N_params)

    # Loop through parameter sets and estimate log10Mmin that yields ng = ng_desired
    for i in range(N_params):
        ng_arr = np.zeros_like(log10Mmin_arr)
        sigma_logM  = sigma_logM_array[i]
        log10M1     = log10M1_array[i]
        kappa       = kappa_array[i]
        alpha       = alpha_array[i]

        # Compute ng for for all log10Mmin
        for j in range(N_Mmin):
            log10Mmin   = log10Mmin_arr[j]

            ng_analytical = compute_ng_analytical(log10Mmin, sigma_logM, log10M1, kappa, alpha, mass_center, dn_dM)
            ng_arr[j] = ng_analytical

        # Interpolate to find log10Mmin that yields ng = ng_desired
        # log10Mmin(n_g) is a very well-behaved function. 
        ng_sorted_idx = np.argsort(ng_arr)

        log10Mmin_spline = UnivariateSpline(ng_arr[ng_sorted_idx], log10Mmin_arr[ng_sorted_idx], k=3, s=0)
        log10Mmin_best_fit_spline = log10Mmin_spline(ng_desired) # Final estiamte 
        log10Mmin_best_fit[i] = log10Mmin_best_fit_spline

        # Compute ng for best fit log10Mmin, to check if it is close to ng_desired
        ng_best_fit_spline = compute_ng_analytical(log10Mmin_best_fit_spline, sigma_logM, log10M1, kappa, alpha, mass_center, dn_dM)
        if test:
            print(f"Best fit [{i:2.0f}]: ng: {ng_best_fit_spline:.4e} | log10Mmin: {log10Mmin_best_fit_spline:.4f} | rel.diff. ng: {np.abs(1.0 - ng_best_fit_spline / ng_desired):.6e}")

    return log10Mmin_best_fit



if __name__ == "__main__":
    estimate_log10Mmin_from_gal_num_density(test=True)
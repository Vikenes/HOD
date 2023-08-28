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

def compute_gal_num_density():

    log10Mmin   = 13.62 # h^-1 Msun
    sigma_logM  = 0.6915
    log10M1     = 14.42 # h^-1 Msun
    kappa       = 0.51
    alpha       = 0.9168
    node_params = [log10Mmin, sigma_logM, log10M1, kappa, alpha]

    cosmology = Cosmology.from_custom(run=0, emulator_data_path=HOD_DATA_PATH)

    redshift = 0.25 
    boxsize  = 2000.0
    
    pos  = np.load(f"{HALO_ARRAYS_PATH}/L1_pos.npy") 
    vel  = np.load(f"{HALO_ARRAYS_PATH}/L1_vel.npy")
    mass = np.load(f"{HALO_ARRAYS_PATH}/L1_mass.npy")

    halocat = HaloCatalogue(
        pos,
        vel,
        mass,
        boxsize,
        conc_mass_model=hmd.concentration.diemer15,
        cosmology=cosmology,
        redshift=redshift,
        )
    N = 250
    mmin = np.log10(np.min(mass))
    mmax = np.log10(np.max(mass))
    mass_edges = np.logspace(mmin, mmax, N)
    dn_dM = halocat.compute_hmf(mass_edges)
    mass_center = 0.5 * (mass_edges[1:] + mass_edges[:-1])

  

    Nc = 0.5 * special.erfc(np.log10(10**node_params[0] / mass_center) / node_params[1])
    mask = mass_center > node_params[3] * 10**node_params[0]
    lambda_values = np.zeros_like(mass_center)
    lambda_values[mask] = ((mass_center[mask] - node_params[3] * 10**node_params[0]) / 10**node_params[2])**node_params[4]

    Ns = Nc * lambda_values
    Ng = Nc + Ns

    integrand = dn_dM * Ng
    dM = mass_center[1] - mass_center[0]



    fff = h5py.File(f"{OUTFILEPATH}/halocat_fiducial.hdf5", "r")
    ng_estimated = fff.attrs['ng']
    nc_estimated = fff.attrs['nc']
    ns_estimated = fff.attrs['ns']

    ng_analytical = simpson(integrand, mass_center, dx=dM)
    nc_analytical = simpson(dn_dM * Nc, mass_center, dx=dM)   
    ns_analytical = simpson(dn_dM * Ns, mass_center, dx=dM) 


    print("       estimated | analytical | difference")
    print(f"ng_bar: {ng_estimated:.4e} | {ng_analytical:.4e} | {np.abs(ng_estimated - ng_analytical):.4e} ")
    print(f"nc_bar: {nc_estimated:.4e} | {nc_analytical:.4e} | {np.abs(nc_estimated - nc_analytical):.4e} ")
    print(f"ns_bar: {ns_estimated:.4e} | {ns_analytical:.4e} | {np.abs(ns_estimated - ns_analytical):.4e} ")
    
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


def estimate_log10Mmin_from_gal_num_density(ng_desired=2.174e-4, flag="train"):


    ### Make halo catalogue 
    # Define cosmology and simparams 
    cosmology = Cosmology.from_custom(run=0, emulator_data_path=HOD_DATA_PATH)
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
    node_params_df = pd.read_csv(f"{HOD_DATA_PATH}/HOD_data/HOD_parameters_{flag}.csv")
    N_params = len(node_params_df)
    log10Mmin_best_fit = np.zeros(N_params)

    # Loop through parameter sets and estimate log10Mmin that yields ng = ng_desired
    for i in range(N_params):
        ng_arr = np.zeros_like(log10Mmin_arr)
        sigma_logM  = node_params_df['sigma_logM'].iloc[i]
        log10M1     = node_params_df['log10M1'].iloc[i]
        kappa       = node_params_df['kappa'].iloc[i]
        alpha       = node_params_df['alpha'].iloc[i]

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

        print(f"Best fit [{i:2.0f}]: ng: {ng_best_fit_spline:.4e} | log10Mmin: {log10Mmin_best_fit_spline:.4f} | rel.diff. ng: {np.abs(1.0 - ng_best_fit_spline / ng_desired):.6e}")





# compute_gal_num_density()
# constrain_ng_fiducial()
estimate_log10Mmin_from_gal_num_density()
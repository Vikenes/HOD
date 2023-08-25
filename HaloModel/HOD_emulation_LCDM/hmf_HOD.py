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
import pandas as pd

import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")

HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
HALO_ARRAYS_PATH = f"{HOD_DATA_PATH}/version0"

OUTFILEPATH = f"{HOD_DATA_PATH}/HOD_data"
dataset_names = ['train', 'test', 'val']

def make_HOD_fiducial():

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
    N = 100
    mmin = np.log10(np.min(mass))
    mmax = np.log10(np.max(mass))
    mass_edges = np.logspace(mmin, mmax, N)
    hmf = halocat.compute_hmf(mass_edges)
    mass_center = 0.5 * (mass_edges[1:] + mass_edges[:-1])
    dn_dM = hmf / mass_center

    galaxy=Galaxy(
            logM_min    = node_params[0],
            sigma_logM  = node_params[1],
            logM1       = node_params[2],
            kappa       = node_params[3],
            alpha       = node_params[4],
    
        ),
    
    Nc_occ = Zheng07Centrals().get_n(halo_mass=mass_center, galaxy=galaxy)
    Ns_occ = Zheng07Sats().get_n(halo_mass=mass_center, galaxy=galaxy)

    print(Nc_occ.shape)
    print(Ns_occ.shape)
    exit()


    print(f"Running Fiducial...", end=" ")
    t0 = time.time()


    maker = HODMaker(
        halo_catalogue=halocat,
        central_occ=Zheng07Centrals(),
        sat_occ=Zheng07Sats(),
        satellite_profile=FixedCosmologyNFW(
            cosmology=halocat.cosmology,
            redshift=redshift,
            mdef="200m",
            conc_mass_model="dutton_maccio14",
            sigmaM=None,
        ),
        galaxy=Galaxy(
            logM_min    = node_params[0],
            sigma_logM  = node_params[1],
            logM1       = node_params[2],
            kappa       = node_params[3],
            alpha       = node_params[4],
    
        ),
    )

    maker()

    Nc_occ = maker.get_central_df()
    print(Nc_occ.keys())
    exit()
    Ns_occ = maker.get_satellite_df()

    galaxy_df = maker.galaxy_df
    galaxy_df_central = galaxy_df[galaxy_df['galaxy_type'] == 'central']
    galaxy_df_satellite = galaxy_df[galaxy_df['galaxy_type'] == 'satellite']
    Ng = len(galaxy_df)
    Nc = len(galaxy_df_central)
    Ns = len(galaxy_df_satellite)
    ng = Ng / (boxsize**3)
    nc = Nc / (boxsize**3)
    ns = Ns / (boxsize**3)
   

    galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["central"], 0)
    galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["satellite"], 1)
    galaxy_df.astype({'galaxy_type': int})
    galaxy_properties = galaxy_df.columns.values.tolist()

    print(f"Finised, took {time.time() - t0:.2f} seconds.")



make_HOD_fiducial()
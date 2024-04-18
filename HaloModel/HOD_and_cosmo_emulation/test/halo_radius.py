import numpy as np 
import h5py 
import time 
from dq import Cosmology
import hmd
from hmd.catalogue import HaloCatalogue
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW
from hmd.populate import HODMaker
from pathlib import Path
import pandas as pd
import sys 

import concurrent.futures

import warnings
warnings.filterwarnings("ignore", message='Astropy cosmology class contains massive neutrinos, which are not taken into account in Colossus.')

D13_BASE_PATH           = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
HOD_PROPERTIES_TO_SAVE = ["x", "y", "z", "s_z"]

dataset_names = ['train', 'test', 'val']
print()
def make_hdf5_files_single_version(
        version:    int  = 0,
        phase:      int  = 0,
        ng_fixed:   bool = True
        ):
    print_every_n = 25 

    simname             = f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}"
    HOD_DATA_PATH       = f"{D13_BASE_PATH}/emulation_files/{simname}"
    OUTFILEPATH         = f"{HOD_DATA_PATH}/HOD_catalogues"
    HOD_PARAMETERS_PATH = f"{HOD_DATA_PATH}/HOD_parameters"
    HALO_ARRAYS_PATH    = f"{HOD_DATA_PATH}/pos_vel_mass_arrays" # directory with pos, vel, mass arrays

    # print(f"Computing for flag={flag}")

    # Set up cosmology and simulation info 
    # Uses the cosmology.dat file found in the HOD_DATA_PATH
    cosmology = Cosmology.from_custom(run=0, emulator_data_path=HOD_DATA_PATH)
    
    # Load pos, vel and mass of halos with mass > 1e12 h^-1 Msun
    pos  = np.load(f"{HALO_ARRAYS_PATH}/L1_pos.npy")  # shape: (N_halos, 3)
    vel  = np.load(f"{HALO_ARRAYS_PATH}/L1_vel.npy")  # shape: (N_halos, 3)
    mass = np.load(f"{HALO_ARRAYS_PATH}/L1_mass.npy") # shape: (N_halos,)

    # print(vel)
    # print(f"{np.max(vel)=} | {np.min(vel)=}")
    # vel /= 2000.0
    # print(f"{np.max(vel)=} | {np.min(vel)=}")

    # # print()
    # exit()


    # Make halo catalogue. 
    # Note: The halo catalogue is independent of the HOD parameters.
    t0 = time.time()
    halocat = HaloCatalogue(
        pos,
        vel,
        mass,
        boxsize         = 2000.0,
        mdef="200c",
        conc_mass_model = hmd.concentration.diemer15,
        cosmology       = cosmology,
        redshift        = 0.25,
        )
    dur = time.time() - t0
    print(f"{dur=:.3f}s")
    halocat_radius = halocat.radius
    conc = halocat.concentration
    print(f"{halocat.concentration.shape=}")
    print(f"{halocat_radius.shape=}")
    # print()
    print(f"{np.min(halocat_radius)=:.3f} | {np.max(halocat_radius)=:.3f}")
    print(f"{np.min(conc)=:.3f} | {np.max(conc)=:.3f}")






make_hdf5_files_single_version(version=0, phase=0)
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

    Path(OUTFILEPATH).mkdir(parents=False, exist_ok=True) # Create directory if it doesn't exist and don't create if parent doesn't exist

    print(f"Making hdf5 files for {simname}...")
    t0_total = time.time()
    for flag in dataset_names:
        # print(f"Computing for flag={flag}")
        filename_suffix = f"{flag}"
        if ng_fixed:
            filename_suffix += "_ng_fixed"
        
        outfname = f"halocat_{filename_suffix}.hdf5"
        OUTFILE = Path((f"{OUTFILEPATH}/{outfname}"))
        if OUTFILE.exists():
            # Prevent overwriting to save time 
            print(f"File {OUTFILE} already exists, skipping...")
            continue

        # Set up cosmology and simulation info 
        # Uses the cosmology.dat file found in the HOD_DATA_PATH
        cosmology = Cosmology.from_custom(run=0, emulator_data_path=HOD_DATA_PATH)
        
        # Make hdf5 file
        fff = h5py.File(f"{OUTFILEPATH}/{outfname}", "w")

        # Store cosmology and simulation info in hdf5 file
        fff.attrs["H0"]                 = float(cosmology.H0.value)
        fff.attrs["Om0"]                = cosmology.Om0
        fff.attrs["Ode0"]               = cosmology.Ode0
        fff.attrs["w0"]                 = cosmology.w0
        fff.attrs["wc0"]                = cosmology.wc0
        fff.attrs["Ob0"]                = cosmology.Ob0
        fff.attrs["Neff"]               = cosmology.Neff
        fff.attrs["lnAs"]               = cosmology.lnAs
        fff.attrs["n_s"]                = cosmology.n_s

        # Load pos, vel and mass of halos with mass > 1e12 h^-1 Msun
        pos  = np.load(f"{HALO_ARRAYS_PATH}/L1_pos.npy")  # shape: (N_halos, 3)
        vel  = np.load(f"{HALO_ARRAYS_PATH}/L1_vel.npy")  # shape: (N_halos, 3)
        mass = np.load(f"{HALO_ARRAYS_PATH}/L1_mass.npy") # shape: (N_halos,)

        # Load HOD parameters
        hod_params_fname = f"{HOD_PARAMETERS_PATH}/HOD_parameters_{filename_suffix}.csv"
        node_params_df   = pd.read_csv(hod_params_fname)

        # Make halo catalogue. 
        # Note: The halo catalogue is independent of the HOD parameters.
        halocat = HaloCatalogue(
            pos,
            vel,
            mass,
            boxsize         = 2000.0,
            conc_mass_model = hmd.concentration.diemer15,
            cosmology       = cosmology,
            redshift        = 0.25,
            )

        # Loop over HOD parameters 
        # Populate halos with galaxies for each HOD parameter set 

        t0_flag = time.time()
        t0_flag_total = time.time()
        for node_idx in range(len(node_params_df)):

            if node_idx % print_every_n == 0 and node_idx != 0 or node_idx == len(node_params_df) - 1:
                print(f"{simname}-{flag}-nodes {node_idx-print_every_n}-{node_idx} complete.", end=" ")
                print(f"Duration: {time.time() - t0_flag:.2f} sec.")
                t0_flag = time.time()
            

            # Store HOD parameters in hdf5 file
            HOD_group                     = fff.create_group(f"node{node_idx}") 
            HOD_group.attrs['log10Mmin']  = node_params_df['log10Mmin'].iloc[node_idx]
            HOD_group.attrs['sigma_logM'] = node_params_df['sigma_logM'].iloc[node_idx]
            HOD_group.attrs['log10M1']    = node_params_df['log10M1'].iloc[node_idx]
            HOD_group.attrs['kappa']      = node_params_df['kappa'].iloc[node_idx]
            HOD_group.attrs['alpha']      = node_params_df['alpha'].iloc[node_idx]

            # Populate halos with galaxies
            maker = HODMaker(
                halo_catalogue      = halocat,
                central_occ         = Zheng07Centrals(),
                sat_occ             = Zheng07Sats(),
                satellite_profile   = FixedCosmologyNFW(
                    cosmology       = halocat.cosmology,
                    redshift        = 0.25,
                    mdef            = "200m",
                    conc_mass_model = "dutton_maccio14",
                    sigmaM          = None,
                    ),
                galaxy=Galaxy(
                    logM_min    = node_params_df['log10Mmin'].iloc[node_idx],
                    sigma_logM  = node_params_df['sigma_logM'].iloc[node_idx],
                    logM1       = node_params_df['log10M1'].iloc[node_idx],
                    kappa       = node_params_df['kappa'].iloc[node_idx],
                    alpha       = node_params_df['alpha'].iloc[node_idx]
                    ),
                )
            maker()

            # Load galaxy catalogue
            galaxy_df           = maker.galaxy_df


            # Add redshift-space distortions to z-coordinate 
            galaxy_df["s_z"] = halocat.apply_redshift_distortion(
                galaxy_df['z'],
                galaxy_df['v_z'],
            )

            # Compute number density of galaxies and store it in hdf5 file
            box_volume = 2000.0**3
            ng = len(galaxy_df) / box_volume
            HOD_group.attrs['ng'] = ng

            # Store galaxy catalogue in hdf5 file
            # galaxy_properties:
            #  - x, y, z, s_z
           
            for prop in HOD_PROPERTIES_TO_SAVE:
                HOD_group.create_dataset(
                    prop, 
                    data = galaxy_df[prop].values,
                    dtype= galaxy_df[prop].dtypes
                    )

        fff.close()
        print(f"{simname}-{flag} complete. Duration: {time.time() - t0_flag_total:.2f} sec.")

    print(f"{simname} complete. Duration: {time.time() - t0_total:.2f} sec.")


def make_hdf5_files_c000_phases_parallel(
        start,
        stop ,
        parallel=True
        ):
    phases = np.arange(start,stop)
    return 
    t000 = time.time()
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(make_hdf5_files_single_version, [0 for i in range(len(phases))], phases)
        
    else:
        for ph in phases:
            make_hdf5_files_single_version(
                version=0,
                phase=ph,
                ng_fixed=True
                )
            
    tot_dur = time.time() - t000
    print(f"Total duration: {tot_dur:.2f} sec")




def make_hdf5_files_emulator_versions_parallel(
        start,
        stop,
        parallel=True
        ):
    versions = np.arange(start, stop)
    return 
    t000 = time.time()
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(make_hdf5_files_single_version, [v for v in versions], [0 for v in versions])
    print(f"Total duration: {time.time() - t000:.2f} sec")

def make_hdf5_files_c001_c004_parallel():
    versions = np.arange(1,5)
    return 
    t000 = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(make_hdf5_files_single_version, [v for v in versions], [0 for v in versions])
    print(f"Total duration: {time.time() - t000:.2f} sec")

def make_hdf5_files_lindergrid_parallel(
        start,
        stop,
        parallel=True
        ):
    versions = np.arange(start, stop)
    return 
    t000 = time.time()
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(make_hdf5_files_single_version, [v for v in versions], [0 for v in versions])
    print(f"Total duration: {time.time() - t000:.2f} sec")

def make_hdf5_files_c000_all_phases(
        N_parallell=5,
        parallel=True):
    for i in range(0, 25, N_parallell):
        make_hdf5_files_c000_phases_parallel(i, i+N_parallell, parallel=parallel)

def make_hdf5_files_all_emulator_versions(
        N_parallel=4,
        parallel=True
        ):
    for i in range(130, 182, N_parallel):
        make_hdf5_files_emulator_versions_parallel(i, i+N_parallel, parallel=parallel)

        
def make_hdf5_files_all_lindergrid_versions(
        N_parallel=3,
        parallel=True
        ):
    for i in range(100, 126, N_parallel):
        make_hdf5_files_lindergrid_parallel(i, i+N_parallel, parallel=parallel)

# make_hdf5_files_c000_all_phases()
# make_hdf5_files_c001_c004_parallel()
# make_hdf5_files_all_emulator_versions(parallel=True)
# make_hdf5_files_all_lindergrid_versions(parallel=True)

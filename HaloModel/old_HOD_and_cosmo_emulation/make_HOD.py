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
import sys 

import concurrent.futures

import warnings
warnings.filterwarnings("ignore", message='Astropy cosmology class contains massive neutrinos, which are not taken into account in Colossus.')

notes = []

D13_BASE_PATH           = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"

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
        redshift  = 0.25 
        boxsize   = 2000.0
        
        # Make hdf5 file
        fff = h5py.File(f"{OUTFILEPATH}/{outfname}", "w")

        # Store cosmology and simulation info in hdf5 file
        fff.attrs["boxsize"]            = boxsize
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
            boxsize,
            conc_mass_model = hmd.concentration.diemer15,
            cosmology       = cosmology,
            redshift        = redshift,
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
                    redshift        = redshift,
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

            # Get number of central and satellite galaxies
            galaxy_df_central   = galaxy_df[galaxy_df['galaxy_type'] == 'central']
            galaxy_df_satellite = galaxy_df[galaxy_df['galaxy_type'] == 'satellite']

            # Compute number density of galaxies and store it in hdf5 file
            box_volume = boxsize**3
            ng = len(galaxy_df) / box_volume
            nc = len(galaxy_df_central) / box_volume
            ns = len(galaxy_df_satellite) / box_volume

            HOD_group.attrs['ng'] = ng
            HOD_group.attrs['nc'] = nc
            HOD_group.attrs['ns'] = ns

            # Convert galaxy type from string to int
            galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["central"], 0)
            galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["satellite"], 1)
            galaxy_df.astype({'galaxy_type': int})

            # Store galaxy catalogue in hdf5 file
            # galaxy_properties:
            #  - host_mass, host_radius, host_concentration
            #  - x, y, z, v_x, v_y, v_z
            #  - galaxy_type
            #  - host_centric_distance, host_id 


            galaxy_properties = galaxy_df.columns.values.tolist()
            
            for prop in galaxy_properties:
                HOD_group.create_dataset(
                    prop, 
                    data = galaxy_df[prop].values,
                    dtype= galaxy_df[prop].dtypes
                    )
            # if node_idx % print_every_n == 0 or node_idx == len(node_params_df) - 1:
                # print(f"complete, duration: {time.time() - t0:.2f} sec.")

        fff.close()
        print(f"{simname}-{flag} complete. Duration: {time.time() - t0_flag_total:.2f} sec.")

    print(f"{simname} complete. Duration: {time.time() - t0_total:.2f} sec.")

def make_hdf5_files_c000_all_phases(
        parallel=True
        ):
    phases = np.arange(0,25)
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


def make_hdf5_files_all_emulator_versions(parallel=True):
    versions = np.arange(130, 182)
    t000 = time.time()
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(make_hdf5_files_single_version, [v for v in versions], [0 for v in versions])
    print(f"Total duration: {time.time() - t000:.2f} sec")

def make_hdf5_files_non_emulator_versions(parallel=True):
    versions = np.arange(1,5)
    versions = np.concatenate((versions, np.arange(100, 127)))
    t000 = time.time()
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(make_hdf5_files_single_version, [v for v in versions], [0 for v in versions])
    print(f"Total duration: {time.time() - t000:.2f} sec")

make_hdf5_files_single_version(
    version=4,
    phase=0,
    ng_fixed=True
    )

# make_hdf5_files_c000_all_phases(parallel=True)
# make_hdf5_files_all_emulator_versions(parallel=True)
# make_hdf5_files_non_emulator_versions(parallel=True)

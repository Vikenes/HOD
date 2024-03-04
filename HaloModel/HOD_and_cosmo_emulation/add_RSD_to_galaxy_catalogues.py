import numpy as np 
import h5py 
import time 
from dq import Cosmology

from hmd.catalogue import GalaxyCatalogue
from pathlib import Path
import pandas as pd

import concurrent.futures 

import warnings
warnings.filterwarnings("ignore", message='Astropy cosmology class contains massive neutrinos, which are not taken into account in Colossus.')

notes = []

D13_BASE_PATH           = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"

dataset_names = ['train', 'test', 'val']

def append_z_pos_to_hdf5_file_single_version(
        version:    int  = 0,
        phase:      int  = 0,
        ng_fixed:   bool = True
        ):

    simname             = f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}"
    HOD_DATA_PATH       = f"{D13_BASE_PATH}/emulation_files/{simname}"
    OUTFILEPATH         = f"{HOD_DATA_PATH}/HOD_catalogues"
    HOD_PARAMETERS_PATH = f"{HOD_DATA_PATH}/HOD_parameters"


    print(f"Making hdf5 files for {simname}...")
    t0_total = time.time()
    for flag in dataset_names:
        # print(f"Computing for flag={flag}")
        filename_suffix = f"{flag}"
        if ng_fixed:
            filename_suffix += "_ng_fixed"
        
        outfname = f"halocat_{filename_suffix}.hdf5"
        OUTFILE = Path((f"{OUTFILEPATH}/{outfname}"))
        if not OUTFILE.exists():
        #     # Prevent overwriting to save time 
            print(f"File {OUTFILE} doesn't exist, aborting...")
            return 

        # Set up cosmology and simulation info 
        # Uses the cosmology.dat file found in the HOD_DATA_PATH
        cosmology = Cosmology.from_custom(run=0, emulator_data_path=HOD_DATA_PATH)
        redshift  = 0.25 
        boxsize   = 2000.0
        
        # Make hdf5 file
        fff = h5py.File(f"{OUTFILEPATH}/{outfname}", "r+")

        # Load HOD parameters
        hod_params_fname = f"{HOD_PARAMETERS_PATH}/HOD_parameters_{filename_suffix}.csv"
        node_params_df   = pd.read_csv(hod_params_fname)

        t0_flag_total = time.time()
        for node_idx in range(len(node_params_df)):

            fff_node = fff[f"node{node_idx}"] 
            
            galaxycat = GalaxyCatalogue(
                pos = np.array([
                    fff_node["x"],
                    fff_node["y"],
                    fff_node["z"],
                    ]).T,
                vel = np.array([
                    fff_node["v_x"],
                    fff_node["v_y"],
                    fff_node["v_z"],
                    ]).T,
                host_mass               = fff_node["host_mass"],
                host_radius             = fff_node["host_radius"],
                host_centric_distance   = fff_node["host_centric_distance"],
                host_id                 = fff_node["host_id"],
                galaxy_type             = fff_node["galaxy_type"],
                boxsize                 = boxsize,
                redshift                = redshift,
                cosmology               = cosmology
            )

            z_pos_df = pd.DataFrame(
                {"s_z": galaxycat.s_pos[:, -1]}
                )
            for key in z_pos_df:
                fff_node.create_dataset(
                    key,
                    data = z_pos_df[key].values,
                    dtype = z_pos_df[key].dtypes
                )
            
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
            executor.map(append_z_pos_to_hdf5_file_single_version, [0 for i in range(len(phases))], phases)
        
    else:
        for ph in phases:
            append_z_pos_to_hdf5_file_single_version(
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
            executor.map(append_z_pos_to_hdf5_file_single_version, [v for v in versions], [0 for v in versions])
    print(f"Total duration: {time.time() - t000:.2f} sec")

def make_hdf5_files_non_emulator_versions(parallel=True):
    versions = np.arange(1,5)
    versions = np.concatenate((versions, np.arange(100, 127)))
    t000 = time.time()
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(append_z_pos_to_hdf5_file_single_version, [v for v in versions], [0 for v in versions])
    print(f"Total duration: {time.time() - t000:.2f} sec")



# make_hdf5_files_c000_all_phases(parallel=True)
# make_hdf5_files_all_emulator_versions(parallel=True)
# make_hdf5_files_non_emulator_versions(parallel=True)

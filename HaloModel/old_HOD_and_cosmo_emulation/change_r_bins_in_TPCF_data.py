import numpy as np 
from pathlib import Path
import pandas as pd
import h5py
import shutil



"""
PREVIOUSLY, TPCF data was saved with the separation bins, r, being the average 
separation between points in each bin, i.e.: 

r_bins = np.array([...])
result = pycorr.TwoPointCorrelationFunction(
    ...,
    edges=r_bins,
    ...
)
r, xi = result(return_sep=True)
-> Save r and xi in hdf5 file 

However, we also want to study how xi/xi_fiducial works for emulation, 
and the therefore needed the same r-values between subsamples. 
When using return_sep=True, the r-values differ slightly between each subsample, 
despite r_bins being fixed. 

THIS SCRIPT makes copies of the TPCF hdf5 files first. 
Computing these are time consuming, so we want to keep the old ones just in case.
Using the copied files, we then make new hdf5 files where we manually set the r-values
and use the xi values from the old files.

Functionality for using the fixed r-values when computing+storing TPCF's is now implemented in
./compute_tpcf.py 
"""




D13_BASE_PATH       = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit")
D13_EMULATION_PATH  = Path(D13_BASE_PATH / "emulation_files")
D13_OUTPATH         = Path(D13_EMULATION_PATH / "TPCF_emulation")



DATASET_NAMES = ["train", "test", "val"]
COSMOLOGY_PARAM_KEYS = ["wb", "wc", "Ol", "lnAs", "ns", "w", "Om", "h", "N_eff"]

# Make list of all simulations containing emulation files 
get_path         = lambda version, phase: Path(D13_EMULATION_PATH / f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}")
Phase_paths      = [get_path(0, ph) for ph in range(25) if get_path(0, ph).is_dir()]
SIMULATION_PATHS = Phase_paths + [get_path(v, 0) for v in range(1,182) if get_path(v, 0).is_dir()]

TPCF_fnames     = [f"TPCF_{flag}_ng_fixed.hdf5" for flag in DATASET_NAMES]


def backup_old_tpcf_files(
        fiducial=False
):
    if fiducial:
        SIMULATION_PATH = Path(D13_EMULATION_PATH / "AbacusSummit_base_c000_ph000")
        OLD_TPCF_DATA_PATH  = Path(f"{SIMULATION_PATH}/TPCF_data")
        NEW_TPCF_DATA_PATH  = Path(f"{SIMULATION_PATH}/TPCF_data/old_r_sep_avg")
        
        NEW_TPCF_DATA_PATH.mkdir(parents=False, exist_ok=True)
        
        TPCF_fiducial_fname = Path("TPCF_fiducial_ng_fixed.hdf5")
        SRC_FILE            = Path(OLD_TPCF_DATA_PATH / TPCF_fiducial_fname)
        DST_FILE            = Path(NEW_TPCF_DATA_PATH / TPCF_fiducial_fname)

        if not DST_FILE.exists():
            shutil.copy(SRC_FILE, DST_FILE)
        else:    
            print(f"File {DST_FILE.name} already exists, skipping...")
            return 
        
    else:

        for SIMULATION_PATH in SIMULATION_PATHS:

            OLD_TPCF_DATA_PATH  = Path(f"{SIMULATION_PATH}/TPCF_data")
            NEW_TPCF_DATA_PATH  = Path(f"{SIMULATION_PATH}/TPCF_data/old_r_sep_avg")
            
            NEW_TPCF_DATA_PATH.mkdir(parents=False, exist_ok=True)

            print(f"Copying TPCF files for {SIMULATION_PATH.name} to {NEW_TPCF_DATA_PATH.name}...")
            

            for TPCF_fname in TPCF_fnames:
                SRC_FILE    = Path(OLD_TPCF_DATA_PATH / TPCF_fname)
                DST_FILE    = Path(NEW_TPCF_DATA_PATH / TPCF_fname) 


                if not DST_FILE.exists():
                    shutil.copy(SRC_FILE, DST_FILE)
                else:    
                    print(f"File {DST_FILE.name} already exists, skipping...")
                    continue



def make_new_tpcf_files(
        fiducial = False,
):
    r_bin_edges = np.concatenate((np.logspace(np.log10(0.01), np.log10(5), 40, endpoint=False),
                                  np.linspace(5.0, 150.0, 75)
                                  ))
    r_bin_centers = (r_bin_edges[1:] + r_bin_edges[:-1]) / 2.0

    if fiducial:
        SIMULATION_PATH     = Path(D13_EMULATION_PATH / "AbacusSummit_base_c000_ph000")
        OLD_TPCF_DATA_PATH  = Path(f"{SIMULATION_PATH}/TPCF_data/old_r_sep_avg")
        if not OLD_TPCF_DATA_PATH.exists():
            print(f"Error: Backup directory {OLD_TPCF_DATA_PATH.name} not found, aborting...")
            raise FileNotFoundError
        
        print(f"Making new TPCF files for {SIMULATION_PATH.name}...")
        NEW_TPCF_DATA_PATH  = Path(f"{SIMULATION_PATH}/TPCF_data")
        TPCF_fname          = Path("TPCF_fiducial_ng_fixed.hdf5")

        # Create outfile 
        old_filename  = Path(OLD_TPCF_DATA_PATH / TPCF_fname)
        new_filename  = Path(NEW_TPCF_DATA_PATH / TPCF_fname)
        
        old_fff = h5py.File(old_filename, "r")
        new_fff = h5py.File(new_filename, "w")

        N_keys = len(old_fff.keys())
        for node_idx in range(N_keys):
            old_fff_node = old_fff[f"node{node_idx}"]
            new_fff_node = new_fff.create_group(f"node{node_idx}")

            old_xi = old_fff_node["xi"][:]

            new_fff_node.create_dataset("r", data=r_bin_centers)
            new_fff_node.create_dataset("xi", data=old_xi)

    else:
        for SIMULATION_PATH in SIMULATION_PATHS:
            OLD_TPCF_DATA_PATH  = Path(f"{SIMULATION_PATH}/TPCF_data/old_r_sep_avg")
            if not OLD_TPCF_DATA_PATH.exists():
                print(f"Error: Backup directory {OLD_TPCF_DATA_PATH.name} not found, aborting...")
                raise FileNotFoundError
            
            print(f"Making new TPCF files for {SIMULATION_PATH.name}...")
            NEW_TPCF_DATA_PATH  = Path(f"{SIMULATION_PATH}/TPCF_data")

            for TPCF_fname in TPCF_fnames:
                
                # Create outfile 
                old_filename  = Path(OLD_TPCF_DATA_PATH / TPCF_fname)
                new_filename  = Path(NEW_TPCF_DATA_PATH / TPCF_fname)
                
                old_fff = h5py.File(old_filename, "r")
                new_fff = h5py.File(new_filename, "w")

                N_keys = len(old_fff.keys())

                for node_idx in range(N_keys):
                    old_fff_node = old_fff[f"node{node_idx}"]
                    new_fff_node = new_fff.create_group(f"node{node_idx}")

                    old_xi = old_fff_node["xi"][:]

                    new_fff_node.create_dataset("r", data=r_bin_centers)
                    new_fff_node.create_dataset("xi", data=old_xi)


                new_fff.close()
                old_fff.close()
        

backup_old_tpcf_files(fiducial=True)
make_new_tpcf_files(fiducial=True)
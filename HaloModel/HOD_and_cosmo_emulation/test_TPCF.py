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

def test_hdf5_edit():
    file = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo/vary_r/TPCF_ng_fixed.hdf5")
    fff = h5py.File(file, "r")
    # fff_data = fff["node0"]
    ff = fff["train"]["AbacusSummit_base_c000_ph000"]["node129"]
    print(ff.keys())
    fff_data = ff["r"]
    print(fff_data[:])


    fff.close()


    exit()
    print("NEW FILE")

    fff_new = h5py.File(file, "a")
    fff_data_new = fff_new["train"]["AbacusSummit_base_c001_ph000"]["node76"]#["xi"]
    
    xi_new = fff_data_new["xi"][4:]
    # print(dir(fff_data_new))
    # exit()
    del fff_data_new["xi"]
    # fff_data_new["xi"][:] = xi_new
    fff_data_new.create_dataset("xi", data=xi_new)
    fff_new.close()
    exit()


def print_r_values(
        fiducial=False,
        sep_avg=False,
):
    if fiducial:
        SIMULATION_PATH     = Path(D13_EMULATION_PATH / "AbacusSummit_base_c000_ph000")
        TPCF_FILE  = Path(f"{SIMULATION_PATH}/TPCF_data/old_r_sep_avg/TPCF_fiducial_ng_fixed.hdf5")
        fff = h5py.File(TPCF_FILE, "r")["node0"]["r"]
        
    else:

        min_r = []
        # print(SIMULATION_PATHS)
        # exit()
        for SIMULATION_PATH in SIMULATION_PATHS:

            TPCF_DATA_PATH  = Path(f"{SIMULATION_PATH}/TPCF_data/old_r_sep_avg")
            
            print(f"Checking version {SIMULATION_PATH.name[-10:]}")
            for TPCF_fname in TPCF_fnames:
                TPCF_FILE    = Path(TPCF_DATA_PATH / TPCF_fname)
                fff = h5py.File(TPCF_FILE, "r")

                for node in fff.keys():
                    r = fff[node]["r"]

                    if np.isnan(r).any():
                        idx = np.where(np.isnan(r))[0][-1]
                        # if idx < 2:
                        #     continue 
                        # print(r[:])
                        # print()
                        # r_ = r[idx+1:]
                        # print(r_)
                        # print()
                        # xi_ = fff[node]["xi"][idx+1:]
                        # print(fff[node]["xi"][:])
                        # print()
                        # print(xi_)
                        # print()
                        print(f"{TPCF_FILE.name=}")
                        print(f"{node=}")
                        print(f"{idx=}")
                        exit()

                        min_r.append(r[idx+1])
             

                    # if np.isnan(r).any():
            # input(f"Finished version {SIMULATION_PATH.name[-10:]}, continue?")
        print(min_r)
        print(np.max(min_r))

test_hdf5_edit()
# print_r_values(fiducial=False)
# a = np.array([1,2,np.nan,4,5,np.nan,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# # Check if any nan values in array
# if np.isnan(a).any():
#     # Print indices of nan values
#     idx = np.where(np.isnan(a))[0][-1]
#     a_low = a[:idx]
#     a = a[idx+1:]
#     print(a)
#     print(a_low)
# else:
#     print("no nan values")

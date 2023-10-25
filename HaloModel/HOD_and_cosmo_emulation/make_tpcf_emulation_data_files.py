import numpy as np 
from pathlib import Path
import pandas as pd
import h5py
import time 

################# 
##### TBD: ######
#################

# Save one TPCF hdf5 file for xi only, and one for xi/xi_fiducial?
# Do all manipulations of the TPCF's in the end when generating csv files? 


"""
Use HOD galaxy catalogues to store TPCF data.
Make hdf5 and csv files with TPCF data for each node in each data set, with columns:
    - sigma_logM
    - alpha
    - kappa
    - log10M1
    - log10Mmin
    - r
    - xi
"""


D13_BASE_PATH       = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit")
D13_EMULATION_PATH  = Path(D13_BASE_PATH / "emulation_files")
D13_OUTPATH         = Path(D13_EMULATION_PATH / "TPCF_emulation")



DATASET_NAMES           = ["train", "test", "val"]
COSMOLOGY_PARAM_KEYS    = ["wb", "wc", "Ol", "lnAs", "ns", "w", "Om", "h", "N_eff"]
HOD_PARAM_KEYS          = ["sigma_logM", "alpha", "kappa", "log10M1", "log10Mmin"]

# Make list of all simulations containing emulation files 
get_path         = lambda version, phase: Path(D13_EMULATION_PATH / f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}")
Phase_paths      = [get_path(0, ph) for ph in range(25) if get_path(0, ph).is_dir()]
SIMULATION_PATHS = Phase_paths + [get_path(v, 0) for v in range(1,182) if get_path(v, 0).is_dir()]

TBD = []
TBD.append("Fix cosmological_parameters.dat that will be saved")
TBD.append("Decide which HOD params to store")


def make_TPCF_HDF_files_arrays_at_fixed_r(
    r_low:      float = 0.6, 
    r_high:     float = 100.0,
    ng_fixed:   bool = True,
    ):

    """
    Create hdf5 files with TPCF data for each node in each data set.
    containing the HOD parameters for each node.
    The HDF5 data is then used to create csv files, which is the data actually used for emulation.

    r_low and r_high are the lower and upper limits of the r interval used for the TPCF data.
    TPCF data outside this interval is noisy, and therefore left out.
    """

    ng_suffix       = "_ng_fixed" if ng_fixed else ""
    TPCF_fnames     = [f"TPCF_{flag}{ng_suffix}.hdf5" for flag in DATASET_NAMES]
    HOD_cat_fnames  = [f"halocat_{flag}{ng_suffix}.hdf5" for flag in DATASET_NAMES]

    out_subdir      = "fixed_r_bins"
    TPCF_OUTPATH    = Path(D13_OUTPATH / out_subdir)
    TPCF_OUTPATH.mkdir(parents=True, exist_ok=True)
    
    # Load fiducial TPCF data
    TPCF_fiducial_fname = f"TPCF_fiducial{ng_suffix}.hdf5"
    TPCF_fiducial_file  = Path(SIMULATION_PATHS[0] / "TPCF_data" / TPCF_fiducial_fname)
    TPCF_fiducial       = h5py.File(TPCF_fiducial_file, "r")
    
    # Load r-bins 
    r_fiducial          = TPCF_fiducial["node0"]["r"][:]

    # if masked:
    #     # Make mask, omitting the noisy data outside r_low and r_high
    #     r_mask_low_limit    = r_fiducial > r_low
    #     r_mask_high_limit   = r_fiducial < r_high
    #     r_mask              = r_mask_low_limit * r_mask_high_limit
    # else:
    #     # Store full data range 
    #     r_mask              = np.ones_like(r_fiducial, dtype=bool)

    # Load fiducial xi data
    xi_fiducial         = TPCF_fiducial["node0"]["xi"][:]


    for TPCF_fname, HOD_cat_fname in zip(TPCF_fnames, HOD_cat_fnames):
        # filename_suffix = f"{flag}{ng_suffix}"
        
        # Create outfile 
        # outfname = f"TPCF_{filename_suffix}.hdf5"
        outfile  = Path(TPCF_OUTPATH / TPCF_fname)
        
        print(f"Storing TPCF's for {outfile.parent.name}/{outfile.name}")

        # Create hdf5 file
        fff     = h5py.File(outfile, "r")

        # Store r and xi_fiducial in hdf5 file
        # r is the same for all nodes, so we only need to store it once
        fff.create_dataset("r", data=r_fiducial)
        fff.create_dataset("xi_fiducial", data=xi_fiducial)

        t0 = time.time()

        # Load data from all simulations
        for SIMULATION_PATH in SIMULATION_PATHS:
            # Create group for each simulation
            # Each group contains the same cosmological parameters
            fff_cosmo  = fff.create_group(SIMULATION_PATH.name)
            
            # Store cosmological parameters
            # cosmo_dict = pd.read_csv(Path(SIMULATION_PATH / "cosmological_parameters.dat"), 
            #                          sep=" "
            #                          ).iloc[0].to_dict()

            # for key in COSMOLOGY_PARAM_KEYS: 
            #     # Store cosmological parameters in hdf5 file
            #     fff_cosmo.attrs[key] = cosmo_dict[key]

            # Load HOD data to access HOD parameters 
            fff_HOD = h5py.File(SIMULATION_PATH / "HOD_catalogues" / HOD_cat_fname, "r")
            N_nodes = len(fff_HOD.keys())
            
            # Load computed TPCF data  
            TPCF_data = h5py.File(SIMULATION_PATH / "TPCF_data" / TPCF_fname, "r")

            # Loop over every HOD parameters set 
            for node_idx in range(N_nodes):
                node_group = fff_cosmo.create_group(f"node{node_idx}")

                # Store HOD parameters in hdf5 file
                HOD_node = fff_HOD[f"node{node_idx}"]

                ### TBD: Store all, or just the actual HOD parameters?
                # for key in HOD_PARAM_KEYS:
                for key in HOD_node.attrs.keys():
                    # print(f"{key}: {HOD_node.attrs[key]}")
                    node_group.attrs[key] = HOD_node.attrs[key] 


                # Load xi data, store it 
                xi_node = TPCF_data[f"node{node_idx}"]["xi"][:]
                node_group.create_dataset("xi", data=xi_node)
            
            fff_HOD.close()

        
        fff.close()
        print("========================================================")
        print(f"Done with {outfile.parent.name}/{outfile.name}. Took {time.time() - t0:.2f} s")
        print("========================================================")
        print()

                   
# make_TPCF_ratio_HDF_files_arrays_at_sliced_r(log_r=True)
# make_TPCF_ratio_HDF_files_arrays_at_sliced_r(log_r=False)


def hdf5_to_csv(
        ng_fixed: bool = True,
        xi_ratio: bool = True,
        log_xi: bool = False,
        log_r: bool = False
        ):
    """
    Make csv files from hdf5 files.
    This is the data actually used for emulation. 
    """
    if xi_ratio:
        datafolder = "xi_over_xi_fiducial"
        if log_r:
            datafolder = "log_r_" + datafolder
    else:
        datafolder = "xi"
        if log_xi:
            datafolder = "log_" + datafolder
        if log_r:
            datafolder = "log_r_" + datafolder
        
    TPCF_HDF5_PATH = Path(D13_EMULATION_PATH / f"TPCF_emulation/{datafolder}")

    if ng_fixed:
        ng_suffix = "_ng_fixed"
    else:
        ng_suffix = ""
    

    for flag in DATASET_NAMES:
        ### Create csv file from hdf5 file 

        print(f"Making csv files in {datafolder} for {flag}...")
        t0 = time.time()
        filename_suffix     = f"{flag}{ng_suffix}"
        TPCF_hdf5_filename  = Path(f"TPCF_{filename_suffix}.hdf5")
    
        fff_TPCF = h5py.File(TPCF_HDF5_PATH / TPCF_hdf5_filename, 'r')

        _out_list = []
        for SIMULATION_PATH in SIMULATION_PATHS:
            fff_TPCF_cosmo = fff_TPCF[SIMULATION_PATH.name]
            cosmo_params_dict = {key: fff_TPCF_cosmo.attrs[key] for key in fff_TPCF_cosmo.attrs.keys()}

            for node_idx in range(len(fff_TPCF_cosmo)):
                fff_TPCF_cosmo_HOD = fff_TPCF_cosmo[f"node{node_idx}"]

                r_key   = "r" if not log_r else "log10r"
                xi_key  = "xi_over_xi_fiducial" if xi_ratio else "xi"
                r_data  = fff_TPCF_cosmo_HOD[r_key][...]
                xi_data = fff_TPCF_cosmo_HOD[xi_key][...]


                HOD_params_dict = {HOD_param: fff_TPCF_cosmo_HOD.attrs[HOD_param] for HOD_param in fff_TPCF_cosmo_HOD.attrs.keys()}
                tot_params_dict = cosmo_params_dict | HOD_params_dict
                df = pd.DataFrame({
                    **tot_params_dict,
                    r_key  : r_data,
                    xi_key : xi_data,
                })
                _out_list.append(df)
            
            # print(f"{SIMULATION_PATH.name} Finished, took {time.time() - t0:.2f} s")

            
        df_all = pd.concat(_out_list)
        df_all.to_csv(
            f'{TPCF_HDF5_PATH}/TPCF_{filename_suffix}.csv',
            index=False,
        )
        fff_TPCF.close()
        print(f"Done. Took {time.time() - t0:.2f} s")
        print()

if len(TBD) != 0:
    print("Warning: Items remaining in TBD:")
    for tbd in TBD:
        print(f" - {tbd}")
    print()
    opt = input("Do you really want to continue??? [y/n] ")
    
    if opt != "y":
        print("Aborting...")
        exit()
    else:
        print("Continuing...")
        print()
        make_TPCF_HDF_files_arrays_at_fixed_r()
        


# hdf5_to_csv()

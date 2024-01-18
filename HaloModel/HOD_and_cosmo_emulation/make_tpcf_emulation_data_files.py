import numpy as np 
from pathlib import Path
import pandas as pd
import h5py
import time 

"""
Make one hdf5 file with TPCF data for all simulations and HOD parameters sets.
For train, test, val.
Create csv files from hdf5 file, storing HOD and cosmological parameters as well. 
"""


D13_BASE_PATH       = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit")
D13_EMULATION_PATH  = Path(D13_BASE_PATH / "emulation_files")
D13_OUTPATH         = Path(D13_EMULATION_PATH / "TPCF_emulation")
D5_BASE_PATH        = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")


DATASET_NAMES           = ["train", "test", "val"]
COSMOLOGY_PARAM_KEYS    = ["wb", "wc", "Ol", "ln1e10As", "ns", "alpha_s", "w", "w0", "wa", "sigma8", "Om", "h", "N_eff"]
HOD_PARAM_KEYS          = ["sigma_logM", "alpha", "kappa", "log10M1", "log10Mmin"]

### Make lists of simulations containing emulation files 

# Get path to emulation data from a simulation, given version and phase
def get_version_path_list(
        version_low:  int  = 0, 
        version_high: int  = 181,
        phase:        bool = False,
        ):
    get_sim_path = lambda version, phase: Path(D13_EMULATION_PATH / f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}")
    if phase:
        version_list = [get_sim_path(0, ph) for ph in range(25) if get_sim_path(0, ph).is_dir()]
    else:
        version_list = [get_sim_path(v, 0) for v in range(version_low, version_high) if get_sim_path(v, 0).is_dir()]
    return version_list 
        

C000_PATHS              = get_version_path_list(phase=True)
C001_C004_PATHS         = get_version_path_list(version_low=1, version_high=5, phase=False)
LIN_DER_GRID_PATHS      = get_version_path_list(version_low=100, version_high=127, phase=False)
BROAD_EMUL_GRID_PATHS   = get_version_path_list(version_low=130, version_high=182, phase=False)
# All simulations
SIMULATION_PATHS        = C000_PATHS + C001_C004_PATHS + LIN_DER_GRID_PATHS + BROAD_EMUL_GRID_PATHS


def train_test_val_paths_split(seed=1998):
    # Use Broad_EMUL_GRID_PATHS and a random half of LIN_DER_GRID_PATHS for training
    # Use other half of LIN_DER_GRID_PATHS for validation
    # Use C000_PATHS and C001_C004_PATHS for testing 
    np.random.seed(seed)
    N_LIN_DER                 = len(LIN_DER_GRID_PATHS)
    LIN_DER_GRID_SHUFFLED     = np.random.permutation(LIN_DER_GRID_PATHS)
    LIN_DER_GRID_TRAIN_PATHS  = list(np.sort(LIN_DER_GRID_SHUFFLED[:N_LIN_DER // 2]))
    LIN_DER_GRID_VAL_PATHS    = list(np.sort(LIN_DER_GRID_SHUFFLED[N_LIN_DER // 2:]))

    train_paths = LIN_DER_GRID_TRAIN_PATHS + BROAD_EMUL_GRID_PATHS
    val_paths   = LIN_DER_GRID_VAL_PATHS
    test_paths  = C000_PATHS + C001_C004_PATHS

    return train_paths, test_paths, val_paths

train_paths, test_paths, val_paths = train_test_val_paths_split()
# Cosmologies to use for training, testing, validation 
SIMULATION_FLAG_PATHS     = {
    "train":    train_paths,
    "test":     test_paths,
    "val":      val_paths,
}


def make_TPCF_HDF_files_arrays_at_varying_r(
    ng_fixed:   bool = True,
    outfname:   str = "TPCF",
    d5:         bool = True,
    ):

    """
    Modifies a copy of existing file. 
    See make_TPCF_HDF_files_arrays_at_fixed_r 
    in the script make_tpcf_emulation_data_fixed_r.py
    for details.

    Make file similar to SIMULATION_PATH/TPCF_data/TPCF_ng_fixed.hdf5,
    but with r-values returned by corrfunc engine, i.e. they VARY for each dataset 

    Make a copy of SIMULATION_PATH/TPCF_data/TPCF_ng_fixed.hdf5, and place it in the directory you want 
    We're only going to store new r-values, and update the xi-datasets if nans exist (empty bin count in corrfunc).  

    """

    # Make output filename, check if it already exists
    ng_suffix       = "_ng_fixed" if ng_fixed else ""
    if d5:
        outpath = Path(D5_BASE_PATH / "vary_r")
    else:
        raise ValueError("Must specify either d5 or d13")
    OUTFILE         = Path(outpath / f"{outfname}{ng_suffix}.hdf5") # /.../TPCF_ng_fixed.hdf5
   
    if OUTFILE.exists():
        # Add option to overwrite file if it exists 
        print(f"Warning: {OUTFILE} already exists.")
        opt = input("Do you want to edit it? [y/n] ")
        if opt != "y":
            print("Aborting...")
            exit()
        else:
            print("Continuing...")
            print()

    # Create output file 
    fff = h5py.File(OUTFILE, "r+")
    
    t0_total = time.time()
    # for TPCF_fname, HOD_cat_fname in zip(TPCF_fnames, HOD_cat_fnames):
    for flag in DATASET_NAMES:
        filename_suffix = f"_{flag}{ng_suffix}.hdf5"
        
        print(f"Storing TPCF's for {flag=}")
        t0 = time.time()

        fff_flag = fff[flag]

        # Load data from all simulations
        for SIMULATION_PATH in SIMULATION_PATHS:
            # Create group for each simulation
            # Each group contains the same cosmological parameters
            # fff_cosmo  = fff_flag.create_group(SIMULATION_PATH.name)
            fff_cosmo  = fff_flag[SIMULATION_PATH.name]

            # Load computed TPCF data  
            # TPCF_data   = h5py.File(SIMULATION_PATH / f"TPCF_data/TPCF{filename_suffix}", "r")
            TPCF_data   = h5py.File(SIMULATION_PATH / f"TPCF_data/old_r_sep_avg/TPCF{filename_suffix}", "r")

            N_nodes     = len(fff_cosmo.keys())

            # Loop over all HOD parameter sets (nodes)
            for node_idx in range(N_nodes):

                # fff_cosmo_node = fff_cosmo.create_group(f"node{node_idx}")
                fff_cosmo_node = fff_cosmo[f"node{node_idx}"]

                # Load xi data, store it 
                xi_node = TPCF_data[f"node{node_idx}"]["xi"][:]
                r_node  = TPCF_data[f"node{node_idx}"]["r"][:]

                # Check if r_node contains nan values
                if np.isnan(r_node).any():
                    # Remove values from r and xi
                    # Keeping only values where above the lowest nan-occurence in r 
                    idx     = np.where(np.isnan(r_node))[0][-1] + 1 # Get lowest index where r is not nan

                    # Slice r and xi arrays 
                    r_node  = r_node[idx:]
                    xi_node = xi_node[idx:] 

                    # Remove old xi dataset, create new one
                    del fff_cosmo_node["xi"]
                    fff_cosmo_node.create_dataset("xi", data=xi_node)

                # Store r data
                fff_cosmo_node.create_dataset("r", data=r_node)

        dur = time.time() - t0        
        print(f"Done with {flag=}. Took {dur//60:.0f}min {dur%60:.0f}sec")
        print()

    dur_tot = time.time() - t0_total
    print(f"Done with all. Took {dur_tot//60:.0f}min {dur_tot%60:.0f}sec")
    fff.close()
                   


def xi_hdf5(
        COSMO_PARAMS_CSV:   list,
        HOD_PARAMS_CSV:     list,
        r_min:              float = 0.6,
        r_max:              float = 100.0,
        ng_fixed:           bool  = True,
        log_r:              bool  = False,
        outdir:             str   = "log10_xi",
        ):
    
    """
    Makes hdf5 files corresponding to the csv files create in xi_hdf5_to_csv().
    These files are used to make the csv files.

    Creating csv files from these hdf5 files is simple, 
    and it allows for simple adjustments to be made to the csv files, i.e. emulator input, if needed.
    
    Also, when analyzing the emulation results, it's much easier to 
    retrieve the correct xi-data from the hdf5 files than from the csv files. 
    Thus, simplifying the process of evaluating the emulator performance. 
    """

    """
    NOTE: The "r-mask" used is constant, since corrfunc returns r-values within fixed bins.
    However, the data loaded in "fff_TPCF" has r and xi values with different shapes for each dataset, 
    since it only contains r-values above which there are no nan values.
    """

    OUTPATH_PARENT = Path(D5_BASE_PATH / "vary_r") # Path to store csv files

    ng_suffix   = "_ng_fixed" if ng_fixed else ""

    # Load hdf5 file with TPCF data
    fff_TPCF    = h5py.File(OUTPATH_PARENT / f"TPCF{ng_suffix}.hdf5", 'r')
    
    
    # Whether to use log10(r) or r
    if log_r:
        outdir += "_log_r"
        r_key  = "log10r"   # Name of r-column in csv file
        # r_out  = np.log10(r_masked)
    else:
        r_key  = "r"        # Name of r-column in csv file
        # r_out  = r_masked

    # xi_key     = "xi_over_xi_fiducial" # Name of xi-column in csv file
    xi_key = "log10xi"
    
    HDF5_OUTPATH = Path(OUTPATH_PARENT / outdir) # Path to store csv files
    HDF5_OUTPATH.mkdir(parents=False, exist_ok=False) # Create directory. Raises error if it already exists. Prevents overwriting files.

    t0_total = time.time()
    for flag in DATASET_NAMES:
        ### Create csv file from hdf5 file 

        print(f"Making hdf5 files for {flag}...")
        t0              = time.time()
    
        OUTFILE_HDF5    = Path(HDF5_OUTPATH / f"TPCF_{flag}{ng_suffix}.hdf5") # /.../TPCF_flag_ng_fixed.hdf5
        if OUTFILE_HDF5.exists():
            print(f"Warning: {OUTFILE_HDF5} already exists.")
            opt = input("Do you want to overwrite it? [y/n] ")
            if opt != "y":
                print("Aborting...")
                exit()
            else:
                print("Continuing...")
                print()

        fff_TPCF_flag   = fff_TPCF[flag] # Load data for train, test, val
        # Create output file
        fff_OUT         = h5py.File(OUTFILE_HDF5, "w") 
        
        # Store common r and xi_fiducial. 
        # fff_OUT.create_dataset(r_key, data=r_out)        
        # fff_OUT.create_dataset("xi_fiducial", data=xi_fiducial)

        for SIMULATION_PATH in SIMULATION_FLAG_PATHS[flag]:

            # Load data for each simulation. 
            fff_TPCF_cosmo      = fff_TPCF_flag[SIMULATION_PATH.name]
            fff_OUT_cosmo       = fff_OUT.create_group(SIMULATION_PATH.name)
            print("Storing data from: ", fff_TPCF_cosmo.name)
            
            # Setup cosmolical parameters dictionary. 
            cosmo_params_dict   = {key: fff_TPCF_cosmo.attrs[key] for key in COSMO_PARAMS_CSV}

            for node_idx in range(len(fff_TPCF_cosmo)):
                # Loop over all HOD parameter sets (nodes) in each simulation.
                fff_TPCF_cosmo_node = fff_TPCF_cosmo[f"node{node_idx}"]
                fff_OUT_cosmo_node  = fff_OUT_cosmo.create_group(f"node{node_idx}")

                # Store HOD parameters in dictionary
                HOD_params_dict = {HOD_param: fff_TPCF_cosmo_node.attrs[HOD_param] for HOD_param in HOD_PARAMS_CSV}

                # Combine dictionaries
                tot_params_dict = cosmo_params_dict | HOD_params_dict
                for key, val in tot_params_dict.items():
                    fff_OUT_cosmo_node.attrs[key] = val

                # Load TPCF data, apply mask 
                r_data  = fff_TPCF_cosmo_node["r"][...]
                r_mask  = (r_data > r_min) & (r_data < r_max)


                xi_data = fff_TPCF_cosmo_node["xi"][...][r_mask]
                xi_out  = np.log10(xi_data)
                r_out   = r_data[r_mask] 

                # Store dataset
                fff_OUT_cosmo_node.create_dataset(xi_key, data=xi_out)
                fff_OUT_cosmo_node.create_dataset(r_key, data=r_out)

        fff_OUT.close()
        dur = time.time() - t0
        print(f"Done. Took {dur//60:.0f}min {dur%60:.2f}sec")
        print()

    fff_TPCF.close()
    dur_tot = time.time() - t0_total
    print(f"Done with all. Took {dur_tot//60:.0f}min {dur_tot%60:.0f}sec")



def xi_hdf5_to_csv(
        ng_fixed:           bool  = True,
        log_r:              bool  = False,
        outdir:             str   = "log10_xi",
        ):
    
    """
    Make three csv files from hdf5 file, one for each train, test, val.
    This is the data that will be used for emulation. 


    Stores every xi/xi_fiducial as a function of r.
    If log_r=True, stores log10(r) instead of r.
    For every xi/xi_fiducial, stores the corresponding cosmological and HOD parameters,
    given by COSMO_PARAMS and HOD_PARAMS.
    These parameters must correspond to the keys in the hdf5 file.

    COSMO_PARAMS can only contain values found in COSMOLOGY_PARAM_KEYS.
    HOD_PARAMS can only contain values found in HOD_PARAM_KEYS.
    """

    OUTPATH_PARENT = Path(D5_BASE_PATH / "vary_r") # Path to store csv files

    ng_suffix    = "_ng_fixed" if ng_fixed else ""

    # Whether to use log10(r) or r
    if log_r:
        outdir += "_log_r"
        r_key  = "log10r"   # Name of r-column in csv file
    else:
        r_key  = "r"        # Name of r-column in csv file

    CSV_OUTPATH = Path(OUTPATH_PARENT / outdir) # Path to store csv files
    CSV_OUTPATH.mkdir(parents=False, exist_ok=True) # Create directory. Raises error if it already exists. Prevents overwriting files.


    t0_total = time.time()
    for flag in DATASET_NAMES:
        ### Create csv file from hdf5 file 


        HDF5_INFILE     = Path(CSV_OUTPATH / f"TPCF_{flag}{ng_suffix}.hdf5")
        if not HDF5_INFILE.exists():
            raise FileNotFoundError(f"{HDF5_INFILE.parent.name}/{HDF5_INFILE.name} does not exist. Cannot make csv files.")
        
        print(f"Making csv files for {flag}...")
        t0              = time.time()
        CSV_OUTFILE     = Path(CSV_OUTPATH / f"TPCF_{flag}{ng_suffix}.csv")
        fff_TPCF_flag   = h5py.File(HDF5_INFILE, "r") # Data for train, test, val

        _out_list = []
        for SIMULATION_PATH in SIMULATION_FLAG_PATHS[flag]:

            # Load data for each simulation. 
            fff_TPCF_cosmo      = fff_TPCF_flag[SIMULATION_PATH.name]
            print("Storing data from: ", fff_TPCF_cosmo.name)

            # Setup cosmolical parameters dictionary. 
            # cosmo_params_dict   = {key: fff_TPCF_cosmo.attrs[key] for key in COSMO_PARAMS_CSV}

            for node_idx in range(len(fff_TPCF_cosmo)):
                # Loop over all HOD parameter sets (nodes) in each simulation.
                fff_TPCF_cosmo_node = fff_TPCF_cosmo[f"node{node_idx}"]

                # Load HOD and cosmo params from hdf5 file, store in dictionary
                tot_params_dict = {key: fff_TPCF_cosmo_node.attrs[key] for key in fff_TPCF_cosmo_node.attrs.keys()}

                # Load xi/xi_fiducial data 
                # print(fff_TPCF_cosmo_node.keys().__iter__().__next__())
                xi_key = fff_TPCF_cosmo_node.keys().__iter__().__next__()
                xi_out = fff_TPCF_cosmo_node[xi_key][:]
                r_out  = fff_TPCF_cosmo_node[r_key][:]
                # Store data in dataframe, append to list
                df = pd.DataFrame({
                    **tot_params_dict,
                    r_key  : r_out,
                    xi_key : xi_out,
                })
                _out_list.append(df)

        df_all          = pd.concat(_out_list)
        
        df_all.to_csv(
            CSV_OUTFILE,
            index=False,
        )
        dur = time.time() - t0
        print(f"Done. Took {dur//60:.0f}min {dur%60:.2f}sec")
        print()
        fff_TPCF_flag.close()

    dur_tot = time.time() - t0_total
    print(f"Done with all. Took {dur_tot//60:.0f}min {dur_tot%60:.0f}sec")







# make_TPCF_HDF_files_arrays_at_varying_r()

COSMO_PARAMS_CSV = ["wb", "wc", "sigma8", "ns", "alpha_s", "N_eff", "w0", "wa"]
HOD_PARAMS_CSV   = ["sigma_logM", "alpha", "kappa", "log10M1", "log10Mmin"]

# xi_hdf5(
#     COSMO_PARAMS_CSV=COSMO_PARAMS_CSV,
#     HOD_PARAMS_CSV=HOD_PARAMS_CSV,
# )

# xi_hdf5_to_csv()
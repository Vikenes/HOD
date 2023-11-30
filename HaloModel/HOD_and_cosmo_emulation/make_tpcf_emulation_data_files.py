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


def make_TPCF_HDF_files_arrays_at_fixed_r(
    ng_fixed:   bool = True,
    outfname:   str = "TPCF",
    d5:         bool = True,
    d13:        bool = False,
    ):

    """
    Create hdf5 files with TPCF data. 
    Create three groups for emulator datasets (train,test,val)
    For dataset, creates one group per simulation, and stores cosmological parameters.
    For each simulation, creates one group per HOD parameter set (node).
    Stores HOD parameters and TPCF data for each node.

    From the resulting HDF5 file, csv files are created using the functions ***_hdf5_to_csv().
    The csv files are the data actually used for emulation.
    Making csv files from the single hdf5 file is very simple.
    Allows for easier adjustments to the data used for emulation, 
    e.g. adjusting r-interval, taking logarithms, using xi/xi_fiducial, etc.

    Also, train/test/val data for all simulations is stored in the same file.
    When making csv files later, only the data from simulations in 'SIMULATION_FLAG_PATHS' 
    are used for train/test/val. 
    This allows for easy adjustments to the train/test/val simulations split.
    """

    # Make output filename, check if it already exists
    ng_suffix       = "_ng_fixed" if ng_fixed else ""
    if d5:
        outpath = D5_BASE_PATH
    elif d13:
        outpath = D13_OUTPATH
    else:
        raise ValueError("Must specify either d5 or d13")
    OUTFILE         = Path(outpath / f"{outfname}{ng_suffix}.hdf5") # /.../TPCF_ng_fixed.hdf5
   
    if OUTFILE.exists():
        # Add option to overwrite file if it exists 
        print(f"Warning: {OUTFILE} already exists.")
        opt = input("Do you want to overwrite it? [y/n] ")
        if opt != "y":
            print("Aborting...")
            exit()
        else:
            print("Continuing...")
            print()
        
    # Create output file 
    fff = h5py.File(OUTFILE, "w")

    # Load fiducial TPCF data
    TPCF_fiducial_fname = f"TPCF_fiducial{ng_suffix}.hdf5"
    TPCF_fiducial_file  = Path(SIMULATION_PATHS[0] / "TPCF_data" / TPCF_fiducial_fname)
    TPCF_fiducial       = h5py.File(TPCF_fiducial_file, "r")
    
    # Load fiducial r-bins and xi data
    r_fiducial          = TPCF_fiducial["node0"]["r"][:]
    xi_fiducial         = TPCF_fiducial["node0"]["xi"][:]

    # Store r and xi_fiducial in hdf5 file
    fff.create_dataset("r", data=r_fiducial) # Same r for all nodes, so only need to store it once
    fff.create_dataset("xi_fiducial", data=xi_fiducial)

    
    t0_total = time.time()
    # for TPCF_fname, HOD_cat_fname in zip(TPCF_fnames, HOD_cat_fnames):
    for flag in DATASET_NAMES:
        filename_suffix = f"_{flag}{ng_suffix}.hdf5"
        
        print(f"Storing TPCF's for {flag=}")
        t0 = time.time()

        # Create group for each dataset (train, test, val)
        fff_flag = fff.create_group(flag)

        # Load data from all simulations
        for SIMULATION_PATH in SIMULATION_PATHS:
            # Create group for each simulation
            # Each group contains the same cosmological parameters
            fff_cosmo  = fff_flag.create_group(SIMULATION_PATH.name)
            
            # Store cosmological parameters, except version number and redshift
            cosmo_dict = pd.read_csv(Path(SIMULATION_PATH / "cosmological_parameters.dat"), 
                                     sep=" "
                                     ).iloc[0].to_dict()
            for key in COSMOLOGY_PARAM_KEYS:
                fff_cosmo.attrs[key] = cosmo_dict[key]

            

            # Load HOD catalogue to access HOD parameters 
            # Use full catalogue rather than the csv files. 
            # Ensures correct correspondence between HOD parameters and TPCF data for each node.  
            fff_HOD     = h5py.File(SIMULATION_PATH / f"HOD_catalogues/halocat{filename_suffix}", "r")

            # Load computed TPCF data  
            TPCF_data   = h5py.File(SIMULATION_PATH / f"TPCF_data/TPCF{filename_suffix}", "r")
            N_nodes     = len(TPCF_data.keys())

            # Loop over all HOD parameter sets (nodes)
            for node_idx in range(N_nodes):
                fff_cosmo_node = fff_cosmo.create_group(f"node{node_idx}")

                # Store HOD parameters in hdf5 file
                for HOD_param_key, HOD_param_val in fff_HOD[f"node{node_idx}"].attrs.items():
                    fff_cosmo_node.attrs[HOD_param_key] = HOD_param_val 
                
                # Load xi data, store it 
                xi_node = TPCF_data[f"node{node_idx}"]["xi"][:]
                fff_cosmo_node.create_dataset("xi", data=xi_node)
            
            fff_HOD.close()

        dur = time.time() - t0        
        print(f"Done with {flag=}. Took {dur//60:.0f}min {dur%60:.0f}sec")
        print()

    dur_tot = time.time() - t0_total
    print(f"Done with all. Took {dur_tot//60:.0f}min {dur_tot%60:.0f}sec")
    fff.close()
                   


def xi_over_xi_fiducial_hdf5(
        COSMO_PARAMS_CSV:   list,
        HOD_PARAMS_CSV:     list,
        r_min:              float = 0.6,
        r_max:              float = 100.0,
        ng_fixed:           bool  = True,
        log_r:              bool  = False,
        outdir:             str   = "xi_over_xi_fiducial",
        ):
    
    """
    Makes hdf5 file version of xi_over_xi_fiducial_hdf5_to_csv().
    Simplifies the process of analyzing the emulation results after training. 
    """

    ng_suffix   = "_ng_fixed" if ng_fixed else ""

    # Load hdf5 file with TPCF data
    fff_TPCF    = h5py.File(D5_BASE_PATH / f"TPCF{ng_suffix}.hdf5", 'r')
    
    # Load fiducial r-bins and xi data. Same r-bins for all simulations. 
    # Only include values where r_min < r < r_max
    r_all       = fff_TPCF["r"][:]
    r_mask      = (r_all > r_min) & (r_all < r_max)
    r_masked    = r_all[r_mask]
    xi_fiducial = fff_TPCF["xi_fiducial"][:][r_mask]
    
    # Whether to use log10(r) or r
    if log_r:
        outdir += "_log_r"
        r_key  = "log10r"   # Name of r-column in csv file
        r_out  = np.log10(r_masked)
    else:
        r_key  = "r"        # Name of r-column in csv file
        r_out  = r_masked

    xi_key     = "xi_over_xi_fiducial" # Name of xi-column in csv file
    
    HDF5_OUTPATH = Path(D5_BASE_PATH / outdir) # Path to store csv files
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
        fff_OUT.create_dataset(r_key, data=r_out)        
        fff_OUT.create_dataset("xi_fiducial", data=xi_fiducial)

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
                xi_data = fff_TPCF_cosmo_node["xi"][...][r_mask]
                xi_out  = xi_data / xi_fiducial 

                # Store dataset
                fff_OUT_cosmo_node.create_dataset(xi_key, data=xi_out)
                # fff_OUT_cosmo_node.create_dataset(r_key, data=r_out)

        fff_OUT.close()
        dur = time.time() - t0
        print(f"Done. Took {dur//60:.0f}min {dur%60:.2f}sec")
        print()

    fff_TPCF.close()
    dur_tot = time.time() - t0_total
    print(f"Done with all. Took {dur_tot//60:.0f}min {dur_tot%60:.0f}sec")



def xi_over_xi_fiducial_hdf5_to_csv(
        ng_fixed:           bool  = True,
        log_r:              bool  = False,
        outdir:             str   = "xi_over_xi_fiducial",
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

    ng_suffix    = "_ng_fixed" if ng_fixed else ""

    # Whether to use log10(r) or r
    if log_r:
        outdir += "_log_r"
        r_key  = "log10r"   # Name of r-column in csv file
    else:
        r_key  = "r"        # Name of r-column in csv file

    CSV_OUTPATH = Path(D5_BASE_PATH / outdir) # Path to store csv files
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
        r_out           = fff_TPCF_flag[r_key][:]

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
                xi_key = fff_TPCF_cosmo_node.keys().__iter__().__next__()
                xi_out = fff_TPCF_cosmo_node[xi_key][:]
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





# make_TPCF_HDF_files_arrays_at_fixed_r()

COSMO_PARAMS_CSV = ["wb", "wc", "sigma8", "ns", "alpha_s", "N_eff", "w0", "wa"]
HOD_PARAMS_CSV   = ["sigma_logM", "alpha", "kappa", "log10M1", "log10Mmin"]

# xi_over_xi_fiducial_hdf5(
#     COSMO_PARAMS_CSV=COSMO_PARAMS_CSV,
#     HOD_PARAMS_CSV=HOD_PARAMS_CSV,
#     outdir="xi_over_xi_fiducial",
# )

# xi_over_xi_fiducial_hdf5_to_csv()
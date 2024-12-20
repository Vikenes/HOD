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
HOD_PARAM_KEYS          = ["sigma_logM", "alpha", "kappa", "log10M1", "log10_ng"]

### Make lists of simulations containing emulation files 



def train_test_val_paths_split(
        split_by_percent:   int = 0.8,
        seed:               int = 42,
        ):
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

    # Use C000_PATHS and C001_C004_PATHS for testing 
    test_paths  = C000_PATHS + C001_C004_PATHS

    np.random.seed(seed)
    if split_by_percent is None:
        # Use Broad_EMUL_GRID_PATHS and a random half of LIN_DER_GRID_PATHS for training
        # Use other half of LIN_DER_GRID_PATHS for validation
        N_LIN_DER                 = len(LIN_DER_GRID_PATHS)
        LIN_DER_GRID_SHUFFLED     = np.random.permutation(LIN_DER_GRID_PATHS)
        LIN_DER_GRID_TRAIN_PATHS  = list(np.sort(LIN_DER_GRID_SHUFFLED[:N_LIN_DER // 2]))
        LIN_DER_GRID_VAL_PATHS    = list(np.sort(LIN_DER_GRID_SHUFFLED[N_LIN_DER // 2:]))
        train_paths = LIN_DER_GRID_TRAIN_PATHS + BROAD_EMUL_GRID_PATHS
        val_paths   = LIN_DER_GRID_VAL_PATHS
    else:
        # Use split_by_percent % of LIN_DER_GRID_PATHS+BROAD_EMUL_GRID_PATHS for training
        # Use the remaining for validation
        assert 0 < split_by_percent < 1, "split_by_percent must be a float between 0 and 1"

        TRAIN_VIABLE        = LIN_DER_GRID_PATHS + BROAD_EMUL_GRID_PATHS
        TRAIN_VIABLE_idx    = np.arange(len(TRAIN_VIABLE))
        N_TRAIN             = int(len(TRAIN_VIABLE) * split_by_percent)
        TRAIN_PATHS_idx     = list(np.sort(np.random.choice(TRAIN_VIABLE_idx, N_TRAIN, replace=False)))
        train_paths         = [TRAIN_VIABLE[i] for i in TRAIN_PATHS_idx]
        val_paths           = [TRAIN_VIABLE[i] for i in TRAIN_VIABLE_idx if i not in TRAIN_PATHS_idx]        


    return train_paths, test_paths, val_paths


train_paths, test_paths, val_paths = train_test_val_paths_split(split_by_percent=0.8)
# Cosmologies to use for training, testing, validation 
SIMULATION_FLAG_PATHS     = {
    "train":    train_paths,
    "test":     test_paths,
    "val":      val_paths,
}




def xi_sliced_hdf5(
        COSMO_PARAMS_CSV:   list,
        HOD_PARAMS_CSV:     list,
        ng_fixed:           bool,
        r_min:              float = 0.6,
        r_max:              float = 100.0,
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
    HDF5_OUTPATH = Path(D5_BASE_PATH / "sliced_r") # Path to store csv and hdf5 files

    ng_suffix     = "_ng_fixed" if ng_fixed else ""
    # Load hdf5 file with TPCF data
    fff_TPCF     = h5py.File(D5_BASE_PATH / f"TPCF_full{ng_suffix}.hdf5", 'r')
    

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
                print()
                return 
            else:
                print("Continuing...")
                print()

        fff_TPCF_flag   = fff_TPCF[flag] # Load data for train, test, val

        with h5py.File(OUTFILE_HDF5, "w") as fff_OUT:
        

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

                    # Load TPCF data, apply r-masking  
                    r_data  = fff_TPCF_cosmo_node["r"][...]
                    r_mask  = (r_data > r_min) & (r_data < r_max)
                    r_data  = r_data[r_mask]

                    # Load TPCF data 
                    xi_data = fff_TPCF_cosmo_node["xi"][...][r_mask]

                    # Check for negative values in xi
                    # Should only be one, in train/AbacusSummit_base_c167_ph000 node 151
                    if (xi_data <= 0).any():
                        """
                        If there are negative values in xi,
                        replace them with the value of the neighbour to the left, except if the zeroth value is negative.

                        To prevent errors if scaing emulator data by taking the log of xi 
                        """
                        print(f"WARNING! NEGATIVE XI FOUND FOR {flag}/{SIMULATION_PATH.name} node {node_idx}...")
                        print(f"  {xi_data=}")
                        print(f"  {np.where(xi_data <= 0)}")
                        input("Continue?")
                        # Replace negative TPCF data with neighbour 
                        xi_neg_indices  = np.where(xi_data < 0)
                        r_data          = np.delete(r_data, xi_neg_indices)
                        xi_data         = np.delete(xi_data, xi_neg_indices)

                    xi_out  = xi_data
                    r_out   = r_data 

                    # Store dataset
                    fff_OUT_cosmo_node.create_dataset("xi", data=xi_out)
                    fff_OUT_cosmo_node.create_dataset("r", data=r_out)

        dur = time.time() - t0
        print(f"Done. Took {dur//60:.0f}min {dur%60:.2f}sec")
        print()

    fff_TPCF.close()
    dur_tot = time.time() - t0_total
    print(f"Done with all. Took {dur_tot//60:.0f}min {dur_tot%60:.0f}sec")



def xi_flags_hdf5(
        COSMO_PARAMS_CSV:   list,
        HOD_PARAMS_CSV:     list,
        sliced_r:           bool,
        ng_fixed:           bool = False,
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
    if sliced_r:
        HDF5_OUTPATH = Path(D5_BASE_PATH / "sliced_r") # Path to store csv and hdf5 files
    else:
        HDF5_OUTPATH = Path(D5_BASE_PATH)

    ng_suffix     = "_ng_fixed" if ng_fixed else ""
    # Load hdf5 file with TPCF data
    

    t0_total = time.time()
    for flag in DATASET_NAMES:
        ### Create csv file from hdf5 file 

        print(f"Making hdf5 files for {flag}...")
        t0              = time.time()
    
        INFILE_HDF5    = Path(HDF5_OUTPATH / f"TPCF_{flag}{ng_suffix}.hdf5") # /.../TPCF_flag_ng_fixed.hdf5
        

        with h5py.File(INFILE_HDF5, "r") as fff:
            N = 0
            avg_r_samples = 0
            for SIMULATION_PATH in SIMULATION_FLAG_PATHS[flag]:

                # Load data for each simulation. 
                fff_cosmo       = fff[SIMULATION_PATH.name]
                for node_idx in range(len(fff_cosmo.keys())):
                    # Loop over all HOD parameter sets (nodes) in each simulation.
                    avg_r_samples += fff_cosmo[f"node{node_idx}"]["r"].shape[0]
                    N += 1
                print(f"{fff_cosmo[f'node{node_idx}']['r'].shape[0]=}")

            avg_r_samples = avg_r_samples // N
            print(f"Average number of r-samples: {avg_r_samples}")
                    

        dur = time.time() - t0
        print(f"Done. Took {dur//60:.0f}min {dur%60:.2f}sec")
        print()

    dur_tot = time.time() - t0_total
    print(f"Done with all. Took {dur_tot//60:.0f}min {dur_tot%60:.0f}sec")


def xi_hdf5_to_csv(
        sliced_r:           bool,
        ng_fixed:           bool,
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
    if sliced_r:
        CSV_OUTPATH = Path(D5_BASE_PATH / "sliced_r") # Path to store csv files
    else:
        CSV_OUTPATH = Path(D5_BASE_PATH)

    ng_suffix    = "_ng_fixed" if ng_fixed else ""

    r_key  = "r"        # Name of r-column in csv file
    xi_key = "xi"       # Name of xi-column in csv file



    t0_total = time.time()
    for flag in DATASET_NAMES:
        outfname_stem = f"TPCF_{flag}{ng_suffix}"
        ### Create csv file from hdf5 file 

        HDF5_INFILE     = Path(CSV_OUTPATH / f"{outfname_stem}.hdf5")
        if not HDF5_INFILE.exists():
            raise FileNotFoundError(f"{HDF5_INFILE.parent.name}/{HDF5_INFILE.name} does not exist. Cannot make csv files.")
        
        CSV_OUTFILE     = Path(CSV_OUTPATH / f"{outfname_stem}.csv")
        if CSV_OUTFILE.exists():
            print(f"Warning: {CSV_OUTFILE} already exists.")
            opt = input("Do you want to overwrite it? [y/n] ")
            if opt != "y":
                print("Aborting...")
                print()
                return 
            else:
                print("Continuing...")
                print()

        print(f"Making csv files for {flag}...")
        t0              = time.time()
        fff_TPCF_flag   = h5py.File(HDF5_INFILE, "r") # Data for train, test, val

        _out_list = []
        for SIMULATION_PATH in SIMULATION_FLAG_PATHS[flag]:

            # Load data for each simulation. 
            fff_TPCF_cosmo      = fff_TPCF_flag[SIMULATION_PATH.name]
            print("Storing data from: ", fff_TPCF_cosmo.name)

            for node_idx in range(len(fff_TPCF_cosmo)):
                # Loop over all HOD parameter sets (nodes) in each simulation.
                fff_TPCF_cosmo_node = fff_TPCF_cosmo[f"node{node_idx}"]

                # Load HOD and cosmo params from hdf5 file, store in dictionary
                tot_params_dict = {key: fff_TPCF_cosmo_node.attrs[key] for key in fff_TPCF_cosmo_node.attrs.keys()}

                # Load xi/xi_fiducial data 
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


COSMO_PARAMS_CSV = ["wb", "wc", "sigma8", "ns", "alpha_s", "N_eff", "w0", "wa"]
HOD_PARAMS_CSV   = ["sigma_logM", "alpha", "kappa", "log10M1", "log10_ng"]

# make_TPCF_hdf5_files_full(ng_fixed=False)

xi_flags_hdf5(
    COSMO_PARAMS_CSV=COSMO_PARAMS_CSV,
    HOD_PARAMS_CSV=HOD_PARAMS_CSV,
    sliced_r=True,
    ng_fixed=False,
)

# xi_sliced_hdf5(
#     COSMO_PARAMS_CSV=COSMO_PARAMS_CSV,
#     HOD_PARAMS_CSV=HOD_PARAMS_CSV,
#     ng_fixed=False,
#     r_min=0.6,
#     r_max=100.0,
# )

# xi_hdf5_to_csv(
#     sliced_r=True,
#     ng_fixed=False,)
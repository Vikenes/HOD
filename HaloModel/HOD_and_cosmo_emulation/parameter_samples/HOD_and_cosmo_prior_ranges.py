import numpy as np 
import pandas as pd
from pathlib import Path
import h5py

EMULPATH        = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")
FIDUCIAL_PATH        = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/fiducial_data")


# HOD and cosmo parameters used in the emulation
HOD_PARAM_NAMES = ["log10M1", "sigma_logM", "kappa", "alpha", "log10_ng"]
COSMO_PARAM_NAMES = ['N_eff', 'alpha_s', 'ns', 'sigma8', 'w0', 'wa', 'wb', 'wc']

DATASET_NAMES   = ["train", "test", "val"]


def store_cosmo_params_prior_range_csv():
    """
    Retrieve all cosmological parameters from all simulations.
    Finds the minimum and maximum value of each parameter.

    Returns a dictionary, with COSMO_PARAM_NAMES as keys
    where each value is a list of [min, max] of the parameter.
    """

    # List to fill with all cosmological parameters 
    outfile = Path("./cosmo_param_limits.csv")
    if outfile.exists():
        print(f"File {outfile} already exists. Load with 'load_cosmo_params_prior_range_csv()'")
        return 

    cosmo_params_all = []
    
    for flag in DATASET_NAMES:
        # Get parameters from train/test/val datasets
        fff             = h5py.File(EMULPATH / f"TPCF_{flag}.hdf5", "r")
        N_sims          = len([fff[simulation] for simulation in fff.keys() if simulation.startswith("AbacusSummit")])

        # Array to fill with all cosmological parameters from one dataset
        cosmo_params    = np.zeros((N_sims, len(COSMO_PARAM_NAMES)))

        for ii, simulation in enumerate(fff.keys()):
            if not simulation.startswith("AbacusSummit"):
                # r and xi_fiducial are datasets in fff, but don't contain any cosmological parameters
                continue

            # Same cosmological parameters for all nodes in one simulation, use node0
            fff_cosmo_node = fff[simulation]["node0"] 
            for jj, param in enumerate(COSMO_PARAM_NAMES):
                # Store each parameter in array
                cosmo_params[ii,jj] = fff_cosmo_node.attrs[param]
        cosmo_params_all.append(cosmo_params)

    # Stack all cosmological parameters from all datasets
    cosmo_params_all    = np.vstack(cosmo_params_all).T

    # Create dictionary with prior range of each parameter    
    cosmo_param_limits = {
        param: [np.min(cosmo_params_all[i]), np.max(cosmo_params_all[i])] for i, param in enumerate(COSMO_PARAM_NAMES)
    }
    df = pd.DataFrame(cosmo_param_limits)
    df.to_csv("cosmo_param_limits.csv", index=False)

def get_cosmo_params_prior_range():
    cosmo_params_file = Path("./cosmo_param_limits.csv")
    if not cosmo_params_file.exists():
        store_cosmo_params_prior_range_csv()
    df = pd.read_csv("cosmo_param_limits.csv").to_dict(orient="list")
    return df 


def get_HOD_params_prior_range():
    """
    All HOD parameters used are previously stored in csv files.
    The prior ranges of all but log10Mmin are given, as they were used to generate the data. 
    log10Mmin is given by fixing the gal.num.dens to a constant value. 

    Thus, only need to retrieve the prior range of log10Mmin from the csv files.

    Returns a dictionary with HOD_PARAM_NAMES as keys
    where each value is a list of [min, max] of the parameter.
    """
    fiducial_HOD_params = get_fiducial_HOD_params()
    log10M1     = fiducial_HOD_params["log10M1"]
    sigma_logM  = fiducial_HOD_params["sigma_logM"]
    kappa       = fiducial_HOD_params["kappa"]
    alpha       = fiducial_HOD_params["alpha"]

    # Create dictionary with prior range of each parameter
    HOD_param_limits = {
        "log10M1"    : [log10M1     * 0.9, log10M1     * 1.1],
        "sigma_logM" : [sigma_logM  * 0.9, sigma_logM  * 1.1],
        "kappa"      : [kappa       * 0.9, kappa       * 1.1],
        "alpha"      : [alpha       * 0.9, alpha       * 1.1],
        "log10_ng"   : [-3.7             ,              -3.2],
    }
    
    return HOD_param_limits


def get_fiducial_HOD_params():
        FIDUCIAL_HOD_params     = pd.read_csv(f"{FIDUCIAL_PATH}/HOD_parameters_fiducial.csv")
        return FIDUCIAL_HOD_params.iloc[0].to_dict()

def get_fiducial_cosmo_params():
    FIDUCIAL_cosmo_params   = pd.read_csv(f"{FIDUCIAL_PATH}/cosmological_parameters.dat", sep=" ")
    return FIDUCIAL_cosmo_params.iloc[0].to_dict()
import numpy as np
import pandas as pd
import time 
from pathlib import Path
import h5py 
import yaml

"""
Creates latex table of HOD and cosmological parameters and their prior ranges.


"""
D13_PATH        = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/")
BASEPATH        = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")
EMULPATH        = Path(BASEPATH / "vary_r")
DATASET_NAMES   = ["train", "test", "val"]

# HOD parameters used in the emulation
HOD_PARAM_NAMES = ["log10Mmin", "log10M1", "sigma_logM", "kappa", "alpha"]

# get the cosmological parameter names from one of the datasets 
with h5py.File(EMULPATH / f"TPCF_test_ng_fixed.hdf5", "r") as f:
    """
    COSMO_PARAM_NAMES is used to retrieve the prior range of each parameter
    and to make the latex label of each parameter.  

    The prior ranges and the latex labels get the same order as COSMO_PARAM_NAMES
    """
    ff                      = f["AbacusSummit_base_c000_ph000"]["node0"]
    COSMO_PARAM_NAMES       = [k for k in ff.attrs.keys() if k not in HOD_PARAM_NAMES]


def get_cosmo_params_prior_range():
    """
    Retrieve all cosmological parameters from all simulations.
    Finds the minimum and maximum value of each parameter.

    Returns a dictionary, with COSMO_PARAM_NAMES as keys
    where each value is a list of [min, max] of the parameter.
    """

    # List to fill with all cosmological parameters 
    cosmo_params_all = []
    
    for flag in DATASET_NAMES:
        # Get parameters from train/test/val datasets
        fff             = h5py.File(EMULPATH / f"TPCF_{flag}_ng_fixed.hdf5", "r")
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

    return cosmo_param_limits

def get_fiducial_params():
        FIDUCIAL_HOD_params     = pd.read_csv(f"{D13_PATH}/fiducial_data/HOD_parameters_fiducial_ng_fixed.csv")
        FIDUCIAL_cosmo_params   = pd.read_csv(f"{D13_PATH}/fiducial_data/cosmological_parameters.dat", sep=" ")
        FIDUCIAL_params         = pd.concat([FIDUCIAL_HOD_params, FIDUCIAL_cosmo_params], axis=1)
        FIDUCIAL_params         = FIDUCIAL_params.iloc[0].to_dict()
        
        # Fiducial_params = [FIDUCIAL_params[param] for param in self.emulator_param_names]
        # print(Fiducial_params)
        return FIDUCIAL_params

def get_fiducial_cosmo_params():
    """
    Retrieve all cosmological parameters from all simulations.
    Finds the minimum and maximum value of each parameter.

    Returns a dictionary, with COSMO_PARAM_NAMES as keys
    where each value is a list of [min, max] of the parameter.
    """

    # List to fill with all cosmological parameters 
    cosmo_params_all = []
    
    for flag in DATASET_NAMES:
        # Get parameters from train/test/val datasets
        fff             = h5py.File(EMULPATH / f"TPCF_{flag}_ng_fixed.hdf5", "r")
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

    return cosmo_param_limits

def get_HOD_params_prior_range():
    """
    All HOD parameters used are previously stored in csv files.
    The prior ranges of all but log10Mmin are given, as they were used to generate the data. 
    log10Mmin is given by fixing the gal.num.dens to a constant value. 

    Thus, only need to retrieve the prior range of log10Mmin from the csv files.

    Returns a dictionary with HOD_PARAM_NAMES as keys
    where each value is a list of [min, max] of the parameter.
    """
    log10M1     = 14.42 # h^-1 Msun
    sigma_logM  = 0.6915 
    kappa       = 0.51 
    alpha       = 0.9168  
    

    log10Mmin_all   = []
    for flag in DATASET_NAMES:
        # Load all log10Mmin values from train/test/val datasets
        df = pd.read_csv(f"./HOD_params_{flag}.csv")
        log10Mmin_all.append(df["log10Mmin"].values)        

    # Stack all log10Mmin values from all datasets
    log10Mmin   = np.hstack(log10Mmin_all)

    # Create dictionary with prior range of each parameter
    HOD_param_limits = {
        "log10Mmin"  : [np.min(log10Mmin), np.max(log10Mmin)],
        "log10M1"    : [log10M1     * 0.9, log10M1     * 1.1],
        "sigma_logM" : [sigma_logM  * 0.9, sigma_logM  * 1.1],
        "kappa"      : [kappa       * 0.9, kappa       * 1.1],
        "alpha"      : [alpha       * 0.9, alpha       * 1.1],
    }
    
    return HOD_param_limits


def make_latex_table(
        # outpath="/uio/hume/student-u74/vetleav/Documents/thesis/ProjectedCorrelationFunctionArticle/tables"
        outpath = "/uio/hume/student-u74/vetleav/Documents/thesis/Masterthesis/masterthesis/tables"
        ):
    
    """
    Create a latex table of HOD and cosmological parameters and their prior ranges.
    Stores the table in outpath/param_priors.tex
    
    The table is formatted as follows:
    -------------------------------------
                Parameter    Prior-range  
    -------------------------------------
     HOD       Latex_name    [min,max]    
               Latex_name    [min,max]    
               ...
    -------------------------------------
    Cosmology  Latex_name    [min,max]
               Latex_name    [min,max]
               ...
    -------------------------------------
    """

    # Column labels of the table
    table_header    = [" ", "Parameter", "Prior range"]
    table_rows      = []

    # Latex labels of HOD and cosmological parameters
    HOD_param_labels = {
        "log10Mmin"     : r"$\log M_\mathrm{min}$",
        "log10M1"       : r"$\log M_1$",
        "sigma_logM"    : r"$\sigma_{\log M}$",
        "kappa"         : r"$\kappa$",
        "alpha"         : r"$\alpha$",
    }
    cosmo_param_labels = {
        "N_eff"     : r"$N_\mathrm{eff}$",
        "alpha_s"   : r"$\dd n_s / \dd \ln k$",
        "ns"        : r"$n_s$",
        "sigma8"    : r"$\sigma_8$",
        "w0"        : r"$w_0$",
        "wa"        : r"$w_a$",
        "wb"        : r"$\omega_b$",
        "wc"        : r"$\omega_\mathrm{cdm}$",
    }

    # Load prior range dicts of HOD and cosmo params 
    HOD_prior_range   = get_HOD_params_prior_range()
    cosmo_prior_range = get_cosmo_params_prior_range()


    # Fill table_rows with HOD parameters
    # for ii, name in enumerate(HOD_param_labels):
    for name in HOD_PARAM_NAMES:
        first_col            = "HOD" if name == HOD_PARAM_NAMES[0] else "" # Add "HOD" to the first row
        label                = HOD_param_labels[name]
        min_prior, max_prior = HOD_prior_range[name]
        prior_range          = f"[{min_prior:.3f}, {max_prior:.3f}]"
        table_rows.append([first_col, label, prior_range])

    # Fill table_rows with cosmological parameters
    for name in COSMO_PARAM_NAMES:
        first_col            = r"\hline Cosmology" if name == COSMO_PARAM_NAMES[0] else "" # Add \hline before "Cosmology" to first row
        label                = cosmo_param_labels[name]
        min_prior, max_prior = cosmo_prior_range[name]
        prior_range          = f"[{min_prior:.3f}, {max_prior:.3f}]"
        table_rows.append([first_col, label, prior_range])

    # Create dataframe and save to latex table.
    # Define caption and label of the table
    df = pd.DataFrame(table_rows, columns=table_header)
    caption = "HOD and cosmological parameters and their prior ranges."
    label = "tab:HOD_and_cosmo_params"

    # Save to latex table
    outfile = Path(f"{outpath}/param_priors.tex")

    # Check if outfile already exists, if so, ask if it should be overwritten
    if outfile.exists():
        print(f"{outfile} already exists.")
        input("Press Enter to overwrite or Ctrl+C to cancel.")

    # Save to latex table
    df.to_latex(
        index=False, 
        escape=False,
        buf=outfile,
        position="h",
        column_format="llr",
        caption=caption,
        label=label,)


def make_latex_table_wide(
        # outpath="/uio/hume/student-u74/vetleav/Documents/thesis/ProjectedCorrelationFunctionArticle/tables"
        outpath = "/uio/hume/student-u74/vetleav/Documents/thesis/Masterthesis/masterthesis/tables"
        ):
    
    """
    Create a latex table of HOD and cosmological parameters and their prior ranges.
    Stores the table in outpath/param_priors.tex
    
    The table is formatted as follows:
    -------------------------------------
                Parameter    Prior-range  
    -------------------------------------
     HOD       Latex_name    [min,max]    
               Latex_name    [min,max]    
               ...
    -------------------------------------
    Cosmology  Latex_name    [min,max]
               Latex_name    [min,max]
               ...
    -------------------------------------
    """


    # Column labels of the table
    table_header    = [" ", "Parameter", "Fiducial value", "Prior range"]
    table_rows      = []

    # Latex labels of HOD and cosmological parameters
    HOD_param_labels = {
        "log10Mmin"     : r"$\log M_\mathrm{min}$",
        "log10M1"       : r"$\log M_1$",
        "sigma_logM"    : r"$\sigma_{\log M}$",
        "kappa"         : r"$\kappa$",
        "alpha"         : r"$\alpha$",
    }
    cosmo_param_labels = {
        "N_eff"     : r"$N_\mathrm{eff}$",
        "alpha_s"   : r"$\dd n_s / \dd \ln k$",
        "ns"        : r"$n_s$",
        "sigma8"    : r"$\sigma_8$",
        "w0"        : r"$w_0$",
        "wa"        : r"$w_a$",
        "wb"        : r"$\omega_b$",
        "wc"        : r"$\omega_\mathrm{cdm}$",
    }

    # Load prior range dicts of HOD and cosmo params 
    HOD_prior_range     = get_HOD_params_prior_range()
    cosmo_prior_range   = get_cosmo_params_prior_range()
    fiducial_params     = get_fiducial_params()


    # Fill table_rows with HOD parameters
    # for ii, name in enumerate(HOD_param_labels):
    for name in HOD_PARAM_NAMES:
        first_col            = "HOD" if name == HOD_PARAM_NAMES[0] else "" # Add "HOD" to the first row
        label                = HOD_param_labels[name]
        min_prior, max_prior = HOD_prior_range[name]
        if fiducial_params[name] == 0:
            fiducial_val = "0.0"
        elif fiducial_params[name] == -1:
            fiducial_val = "-1.0"
        else:
            fiducial_val = f"{fiducial_params[name]:.3f}"
        prior_range          = f"[{min_prior:.3f}, {max_prior:.3f}]"
        # min_prior            = f"{min_prior:.3f}"
        # max_prior            = f"{max_prior:.3f}"
        table_rows.append([first_col, label, fiducial_val, prior_range])

    # Fill table_rows with cosmological parameters
    for name in COSMO_PARAM_NAMES:
        first_col            = r"\hline Cosmology" if name == COSMO_PARAM_NAMES[0] else "" # Add \hline before "Cosmology" to first row
        label                = cosmo_param_labels[name]
        min_prior, max_prior = cosmo_prior_range[name]
        if fiducial_params[name] == 0:
            fiducial_val = "0.0"
        elif fiducial_params[name] == -1:
            fiducial_val = "-1.0"
        else:
            fiducial_val = f"{fiducial_params[name]:.3f}"
        prior_range          = f"[{min_prior:.3f}, {max_prior:.3f}]"
        # min_prior            = f"{min_prior:.3f}"
        # max_prior            = f"{max_prior:.3f}"
        # table_rows.append([first_col, label, min_prior, max_prior])
        table_rows.append([first_col, label, fiducial_val, prior_range])


    # Create dataframe and save to latex table.
    # Define caption and label of the table
    df = pd.DataFrame(table_rows, columns=table_header)
    caption = "HOD and cosmological parameters and their prior ranges."
    label = "tab:HOD_and_cosmo_params"

    # Save to latex table
    outfile = Path(f"{outpath}/param_priors.tex")

    # Check if outfile already exists, if so, ask if it should be overwritten
    if outfile.exists():
        print(f"{outfile} already exists.")
        input("Press Enter to overwrite or Ctrl+C to cancel.")

    # Save to latex table
    df.to_latex(
        index=False, 
        escape=False,
        buf=outfile,
        position="h",
        column_format=" X X X rX ",
        caption=caption,
        label=label,)

def make_priors_config_file():

    hod = get_HOD_params_prior_range()
    cosmo = get_cosmo_params_prior_range()

    # Change dictionary entries to dtype list(float, float)
    hod = {key: [float(val[0]), float(val[1])] for key, val in hod.items()}
    cosmo = {key: [float(val[0]), float(val[1])] for key, val in cosmo.items()}

    # priors = {
    #     "HOD": hod, 
    #     "cosmology": cosmo
    #     }
    
    priors = hod | cosmo
    
    outpath = Path("/mn/stornext/d5/data/vetleav/HOD_AbacusData/covariance_data_fiducial")
    outfile = Path(outpath / "priors_config.yaml")
    with open(outfile, "w") as f:
        yaml.dump(priors, f, default_flow_style=False)
    
make_latex_table_wide()
    
# make_priors_config_file()
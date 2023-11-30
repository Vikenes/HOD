import numpy as np
import pandas as pd
import time 
from pathlib import Path
import h5py 

"""
Creates latex table of HOD and cosmological parameters and their prior ranges.


"""

BASEPATH        = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")
EMULPATH        = Path(BASEPATH / "xi_over_xi_fiducial")
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

def get_cosmo_param_names_latex():
    """
    Translate the cosmo param names to latex labels.
    The function is manually adjusted to fit the ones used in the EMULPATH data.
    If others are used in the future, this function needs to be updated.  

    Returns a list of latex labels for each parameter with same order as COSMO_PARAM_NAMES.
    """
    COSMO_PARAM_NAMES_LABELS = []

    for name in COSMO_PARAM_NAMES:
        if len(name) == 2 and name[0] == "w":
            if name == "wc":
                # CDM density: wc -> \omega_cdm
                gg = rf"\omega_\mathrm{{cdm}}"
            elif name == "wb":
                # Baryon density: wb -> \omega_b
                gg = rf"\omega_\mathrm{{b}}"
            else:
                # EoS parameter
                gg = f"{name[0]}_{name[1]}"

        elif name=="N_eff":
            g1, g2 = name.split("_")
            gg = rf"{g1}_\mathrm{{{g2}}}"
        elif name=="alpha_s":
            # Running of the spectral index. Use dn_s/dlnk instead of alpha_s
            # alpha's are used for HOD parameters, avoid confusion
            gg = rf"\dd n_s / \dd \ln k"

        
        elif name[-1] == "8":
            # sigma8 -> \sigma_8
            gg = rf"\{name[:-1]}_{name[-1]}"

        COSMO_PARAM_NAMES_LABELS.append(rf"${gg}$")
    return COSMO_PARAM_NAMES_LABELS


def get_cosmo_params_prior_range():
    """
    Retrieve all cosmological parameters from all simulations.
    Finds the minimum and maximum value of each parameter.

    Returns an array of shape (8,2), where the second axis is the [min,max] values of each parameter.
    The order of the parameters is the same as in COSMO_PARAM_NAMES.
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

    # Prior ranges of each param, shape (8,2)
    cosmo_param_limits  = np.array(
        [(np.min(cosmo_params_all[i]), np.max(cosmo_params_all[i])) for i in range(8)]
    )

    return cosmo_param_limits


def get_HOD_param_names_latex():
    """
    Translate the HOD param names to latex labels.
    The function is manually adjusted to fit the ones used in the EMULPATH data.
    If others are used in the future, this function needs to be updated.  

    Returns a list of latex labels for each parameter with same order as HOD_PARAM_NAMES.
    """
    HOD_PARAM_NAMES_LABELS = []
    
    for name in HOD_PARAM_NAMES:
        if not "log" in name:
            # either "alpha" or "kappa"
            gg = rf"\{name}" 
        elif name == "sigma_logM":
            gg = rf"\sigma_{{\log M}}"
        else:
            # log10Mmin/log10M1 -> \log_{10} M_min/M_1 
            gg = rf"\log_{{10}}M_\mathrm{{{name[6:]}}}"

        HOD_PARAM_NAMES_LABELS.append(f"${gg}$")
    return HOD_PARAM_NAMES_LABELS


def get_HOD_params_prior_range():
    """
    All HOD parameters used are previously stored in csv files.
    The prioir ranges of all but log10Mmin are given, as they were used to generate the data. 
    log10Mmin is given by fixing the gal.num.dens to a constant value. 

    Thus, only need to retrieve the prior range of log10Mmin from the csv files.

    Return an array of shape (5,2), where the second axis is the [min,max] values of each parameter.
    The order of the parameters is the same as in HOD_PARAM_NAMES.
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
    
    HOD_param_limits = np.array([
        [np.min(log10Mmin), np.max(log10Mmin)],
        [log10M1    * 0.9,  log10M1     * 1.1],
        [sigma_logM * 0.9,  sigma_logM  * 1.1],
        [kappa      * 0.9,  kappa       * 1.1],
        [alpha      * 0.9,  alpha       * 1.1],
    ])
    return HOD_param_limits



def make_latex_table(
        outpath="/uio/hume/student-u74/vetleav/Documents/thesis/ProjectedCorrelationFunctionArticle/tables"
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
               Latex_name    [min,max]    
               ...
    -------------------------------------
    Cosmology  Latex_name    [min,max]
               Latex_name    [min,max]
               Latex_name    [min,max]
               ...
    -------------------------------------
    """

    # Column labels of the table
    table_header    = [" ", "Parameter", "Prior range"]
    table_rows      = []

    # Load HOD latex labels and prior ranges
    HOD_param_labels = get_HOD_param_names_latex()
    HOD_param_limits = get_HOD_params_prior_range()
    
    # Fill table_rows with HOD parameters
    for ii, name in enumerate(HOD_param_labels):
        first_row   = "HOD" if ii == 0 else "" # Add "HOD" to the first row
        prior_range = f"[{HOD_param_limits[ii][0]:.3f}, {HOD_param_limits[ii][1]:.3f}]"
        table_rows.append([first_row, name, prior_range])

    # Load cosmological latex labels and prior ranges
    cosmo_param_labels = get_cosmo_param_names_latex()
    cosmo_param_limits = get_cosmo_params_prior_range()

    # Fill table_rows with cosmological parameters
    for ii, name in enumerate(cosmo_param_labels):
        first_row   = r"\hline Cosmology" if ii == 0 else "" # Add \hline before "Cosmology" to first row
        prior_range = f"[{cosmo_param_limits[ii][0]:.3f}, {cosmo_param_limits[ii][1]:.3f}]"
        table_rows.append([first_row, name, prior_range])

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
        column_format="llcr",
        caption=caption,
        label=label,)


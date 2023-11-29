import numpy as np
import pandas as pd
import time 
from pathlib import Path
import h5py 

# DATAPATH = Path("mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation/HOD_parameters")
# D13_BASE_PATH       = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit")
# D13_EMULATION_PATH  = Path(f"{D13_BASE_PATH}/emulation_files")
# D5_EMULATION_PATH   = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")


BASEPATH        = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")
EMULPATH        = Path(BASEPATH / "xi_over_xi_fiducial")
DATASET_NAMES   = ["train", "test", "val"]

HOD_PARAM_NAMES = ["log10Mmin", "log10M1", "sigma_logM", "kappa", "alpha"]
with h5py.File(BASEPATH / f"TPCF_ng_fixed.hdf5", "r") as f_:
    N_sims_tot = len(f_["train"].keys())

with h5py.File(EMULPATH / f"TPCF_test_ng_fixed.hdf5", "r") as f:
    ff                      = f["AbacusSummit_base_c000_ph000"]["node0"]
    COSMO_PARAM_NAMES       = [k for k in ff.attrs.keys() if k not in HOD_PARAM_NAMES]
    COSMO_PARAMS_FIDUCIAL   = {k: ff.attrs[k] for k in COSMO_PARAM_NAMES}

def get_cosmo_param_names_latex():
    COSMO_PARAM_NAMES_LABELS = []
    for name in COSMO_PARAM_NAMES:
        if len(name) == 2:
            if name == "wc":
                gg = rf"\omega_\mathrm{{cdm}}"
            elif name == "wb":
                gg = rf"\omega_\mathrm{{b}}"
            else:
                gg = f"{name[0]}_{name[1]}"

        elif name=="N_eff":
            g1, g2 = name.split("_")
            gg = rf"{g1}_\mathrm{{{g2}}}"
        elif name=="alpha_s":
            gg = rf"\dd n_s / \dd \ln k"

        
        elif name[-1] == "8":
            gg = rf"\{name[:-1]}_{name[-1]}"

        COSMO_PARAM_NAMES_LABELS.append(rf"${gg}$")
    return COSMO_PARAM_NAMES_LABELS

def get_cosmo_params_prior_range():

    cosmo_params_all = []
    
    for flag in DATASET_NAMES:
        fff             = h5py.File(EMULPATH / f"TPCF_{flag}_ng_fixed.hdf5", "r")
        N_sims          = len([fff[simulation] for simulation in fff.keys() if simulation.startswith("AbacusSummit")])
        cosmo_params    = np.zeros((N_sims, len(COSMO_PARAM_NAMES)))

        for ii, simulation in enumerate(fff.keys()):
            if not simulation.startswith("AbacusSummit"):
                continue
            fff_cosmo_node = fff[simulation]["node0"] 
            for jj, param in enumerate(COSMO_PARAM_NAMES):
                cosmo_params[ii,jj] = fff_cosmo_node.attrs[param]
        cosmo_params_all.append(cosmo_params)


    cosmo_params_all    = np.vstack(cosmo_params_all).T
    cosmo_param_limits  = np.array(
        [(np.min(cosmo_params_all[i]), np.max(cosmo_params_all[i])) for i in range(8)]
    )

    return cosmo_param_limits

def get_HOD_param_names_latex():
    HOD_PARAM_NAMES_LABELS = []
    
    for name in HOD_PARAM_NAMES:
        if not "log" in name:
            gg = rf"\{name}" 
        elif name == "sigma_logM":
            gg = rf"\sigma_{{\log M}}"
        else:
            gg = rf"\log_{{10}}M_\mathrm{{{name[6:]}}}"

        HOD_PARAM_NAMES_LABELS.append(f"${gg}$")
    return HOD_PARAM_NAMES_LABELS

def get_HOD_params_prior_range():

    log10M1     = 14.42 # h^-1 Msun
    sigma_logM  = 0.6915 
    kappa       = 0.51 
    alpha       = 0.9168  
    

    log10Mmin_all   = []
    for flag in DATASET_NAMES:
        df = pd.read_csv(f"./HOD_params_{flag}.csv")
        log10Mmin_all.append(df["log10Mmin"].values)        
    log10Mmin   = np.hstack(log10Mmin_all)
    
    HOD_param_limits = np.array([
        [np.min(log10Mmin), np.max(log10Mmin)],
        [log10M1    * 0.9,  log10M1     * 1.1],
        [sigma_logM * 0.9,  sigma_logM  * 1.1],
        [kappa      * 0.9,  kappa       * 1.1],
        [alpha      * 0.9,  alpha       * 1.1],
    ])
    return HOD_param_limits
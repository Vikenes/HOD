import h5py 
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd 


import sys
sys.path.append("/uio/hume/student-u74/vetleav/Documents/thesis/HOD/HaloModel/HOD_and_cosmo_emulation")


def store_log10Mmin_MGGLAM(outfile, params):
    from numdens_HOD import estimate_log10Mmin_from_gal_num_density_MGGLAM

    sigma_logM, log10M1, kappa, alpha = params

    log10Mmin   = estimate_log10Mmin_from_gal_num_density_MGGLAM(
        sigma_logM_array    = sigma_logM,
        log10M1_array       = log10M1,
        kappa_array         = kappa,
        alpha_array         = alpha,
        )
    if Path(outfile).exists() or Path(f"{outfile}.npy").exists():
        raise FileExistsError(f"{outfile} already exists. Change something...")
    
    np.save(
        "log10Mmin_MGGLAM",
        log10Mmin,
    )

def load_log10Mmin_MGGLAM(infile="log10Mmin_MGGLAM"):
    if Path(infile).exists():
        return np.load(infile)
    elif Path(f"{infile}.npy").exists():
        return np.load(f"{infile}.npy")
    else:
        raise FileNotFoundError(f"{infile} does not exist. Save file first.")
    
def get_log10Mmin_MGGLAM_mean(infile="log10Mmin_MGGLAM"):
    log10Mmin = load_log10Mmin_MGGLAM(infile)
    return np.mean(log10Mmin)

def store_cosmo_params():
    from astropy.cosmology import Planck15, wCDM

    d13_fiducial_data_path = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/fiducial_data")
    outfile = Path(d13_fiducial_data_path / "MGGLAM/cosmological_parameters.dat")
    if outfile.exists():
        raise FileExistsError(f"{outfile} already exists. Change something...")

    h        = Planck15.h
    Om0      = Planck15.Om0
    Ol       = Planck15.Ode0
    ln1e10As = np.log(2.0830e-9 * 1.0e10)
    ns       = Planck15.meta["n"]
    alpha_s  = 0.0 
    w        = -1.0
    w0       = -1.0 
    wa       = 0.0
    sigma8   = Planck15.meta["sigma8"]
    wc0      = Planck15.Odm0*h**2
    Ob0      = Planck15.Ob0
    Tcmb0    = Planck15.Tcmb0.value
    Neff     = Planck15.Neff

    df = pd.DataFrame({
        'version'   : 0,
        'redshift'  : 0.25,
        'wb'        : Ob0 * h**2,
        'wc'        : wc0,
        'Ol'        : Ol,
        'ln1e10As'  : ln1e10As,
        'ns'        : ns,
        'alpha_s'   : alpha_s,
        'w'         : w,
        'w0'        : w0,
        'wa'        : wa,
        'sigma8'    : sigma8,
        'Om'        : Om0,
        'h'         : h,
        'N_eff'     : Neff
    }, index=[0])

    df.to_csv(outfile, index=False, sep=" ")


# HOD_parameters 
HOD_PARAMS_PATH  = f"/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files"
HOD_PARAMS          = pd.read_csv(f"{HOD_PARAMS_PATH}/AbacusSummit_base_c000_ph000/HOD_parameters/HOD_parameters_fiducial_ng_fixed.csv")

sigma_logM  = HOD_PARAMS["sigma_logM"][0]#0.6915 
log10M1     = HOD_PARAMS["log10M1"][0]#14.42 # h^-1 Msun
kappa       = HOD_PARAMS["kappa"][0]#0.51 
alpha       = HOD_PARAMS["alpha"][0]#0.9168  
log10Mmin   = np.mean(load_log10Mmin_MGGLAM("log10Mmin_MGGLAM"))

filename = "/mn/stornext/d8/data/chengzor/MGGLAMx100/GR_halocat_z0.25.hdf5"

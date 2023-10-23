import numpy as np 
import pandas as pd
import time
from pathlib import Path

from numdens_HOD import estimate_log10Mmin_from_gal_num_density

D13_BASE_PATH       = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
D13_EMULATION_PATH  = f"{D13_BASE_PATH}/emulation_files"


def make_csv_file(
        fix_ng:     bool  = True,
        ng_desired: float = 2.174e-4, # h^3 Mpc^-3
        version:    int   = 0,
        phase:      int   = 0,
        ):
    """
    Create parameter datasets with LHS sampler.
    generates   num_train nodes for training, 
                num_test nodes for testing,
                num_val nodes for validation.

    If fix_ng=True, log10Mmin is estimated such that the galaxy number density becomes ng_desired.
    """

    print("Making csv file for fiducial HOD parameters.")
    ### FIDUCIAL VALUES. From https://arxiv.org/pdf/2208.05218.pdf, Table 3  ###
    sigma_logM  = 0.6915 
    log10M1     = 14.42 # h^-1 Msun
    kappa       = 0.51 
    alpha       = 0.9168  


    if not fix_ng:
        # Vary log10Mmin by +/- 10% around fiducial value
        log10Mmin           = 13.62 # h^-1 Msun

    # Create parameter files. 
    version_str = str(version).zfill(3)
    phase_str   = str(phase).zfill(3)
    simname     = f"AbacusSummit_base_c{version_str}_ph{phase_str}"

    # Check if simulation data exists, if not, raise error
    # Prevents creation of files for non-existing simulations!
    if not Path(f"{D13_BASE_PATH}/{simname}").exists(): 
        print(f"Error: simulation '{simname}' does not exist. ")
        raise FileNotFoundError
    
    HOD_PARAMETERS_PATH = Path(f"{D13_EMULATION_PATH}/{simname}/HOD_parameters")
    HOD_PARAMETERS_PATH.mkdir(parents=True, exist_ok=True)

    fname = f"HOD_parameters_fiducial"
    if fix_ng:
        fname += "_ng_fixed" 
    fname   = Path(f"{fname}.csv")
    outfile = Path(HOD_PARAMETERS_PATH / fname)
    if outfile.exists():
        print(f"File {outfile} already exists. Skipping.")
        return 
    
    # Sample parameters
    if fix_ng:
        start       = time.time() 
        log10Mmin   = estimate_log10Mmin_from_gal_num_density(
            sigma_logM_array    = [sigma_logM],
            log10M1_array       = [log10M1],
            kappa_array         = [kappa],
            alpha_array         = [alpha],
            ng_desired          = ng_desired,
            test                = False,
            )[0]
        
        print(f"Estimating log10Mmin for fiducial dataset took {time.time() - start:.2f} seconds.")
    if type(log10Mmin) == np.ndarray:
        log10Mmin = log10Mmin[0]
    # Save parameters to csv file
    df = pd.DataFrame({
        'log10Mmin'     : [log10Mmin],
        'sigma_logM'    : [sigma_logM],
        'log10M1'       : [log10M1],
        'kappa'         : [kappa],
        'alpha'         : [alpha],
    })
    
    # Save to csv file
    df.to_csv(
        outfile,
        index=False
    )
    print(f"Saved {outfile}")


# make_csv_file()
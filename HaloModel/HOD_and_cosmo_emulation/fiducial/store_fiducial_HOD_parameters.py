import numpy as np 
import pandas as pd
import time
from pathlib import Path

import sys 
sys.path.append("../")
from numdens_HOD import get_log10Mmin_from_varying_log_ng

D13_BASE_PATH = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files")
OUTPATH       = Path(D13_BASE_PATH / "fiducial_data")

def store_fiducial_csv():
    """
    Make csv file for fiducial HOD parameters for c000_ph000.
    """

    print("Making csv file for fiducial HOD parameters.")
    ### FIDUCIAL VALUES. From https://arxiv.org/pdf/2208.05218.pdf, Table 3  ###
    sigma_logM  = 0.6915 
    log10M1     = 14.42 # h^-1 Msun
    kappa       = 0.51 
    alpha       = 0.9168  
    log10_ng    = -3.45 # h^3 Mpc^-3

    outfile = Path(OUTPATH / "HOD_parameters_fiducial.csv")

    log10Mmin = get_log10Mmin_from_varying_log_ng(
        np.asarray(sigma_logM), 
        np.asarray(log10M1), 
        np.asarray(kappa), 
        np.asarray(alpha), 
        np.asarray(log10_ng),
        version = 0,
        phase   = 0,
    )[0]

    df = pd.DataFrame({
        'log10Mmin'     : [log10Mmin],
        'log10M1'       : [log10M1],
        'sigma_logM'    : [sigma_logM],
        'kappa'         : [kappa],
        'alpha'         : [alpha],
        'log10_ng'      : [log10_ng],
    })
    
    # Save to csv file
    df.to_csv(
        outfile,
        index=False
    )
    print(f"Saved {outfile}")


def store_all_c000_phases_csv():
    """
    Make csv file for fiducial HOD parameters for all c000 phases.
    """

    ### FIDUCIAL VALUES. From https://arxiv.org/pdf/2208.05218.pdf, Table 3  ###
    sigma_logM_array  = np.asarray(0.6915) 
    log10M1_array     = np.asarray(14.42) # h^-1 Msun
    kappa_array       = np.asarray(0.51) 
    alpha_array       = np.asarray(0.9168)  
    log10_ng_array    = np.asarray(-3.45) # h^3 Mpc^-3



    for i in range(25):
        simname = f"AbacusSummit_base_c000_ph{str(i).zfill(3)}"
        outfile = Path(D13_BASE_PATH / f"{simname}/HOD_parameters/HOD_parameters_fiducial.csv")
        log10Mmin_array = get_log10Mmin_from_varying_log_ng(
            sigma_logM_array, 
            log10M1_array, 
            kappa_array, 
            alpha_array, 
            log10_ng_array,
            version = 0,
            phase   = i,
        )[0]

        df = pd.DataFrame({
            'log10Mmin'     : [log10Mmin_array],
            'log10M1'       : [float(log10M1_array)],
            'sigma_logM'    : [float(sigma_logM_array)],
            'kappa'         : [float(kappa_array)],
            'alpha'         : [float(alpha_array)],
            'log10_ng'      : [float(log10_ng_array)],
        })
        
        # Save to csv file
        df.to_csv(
            outfile,
            index=False
        )
        print(f"Saved {outfile}")

store_all_c000_phases_csv()
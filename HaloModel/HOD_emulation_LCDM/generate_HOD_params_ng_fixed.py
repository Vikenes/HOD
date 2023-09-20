import numpy as np
import pandas as pd
import time 
from smt.sampling_methods import LHS
from pathlib import Path
import hmd 
# from typing import 

from numdens_HOD import estimate_log10Mmin_from_gal_num_density

# DATAPATH = Path("mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation/HOD_parameters")
HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
HALO_ARRAYS_PATH = f"{HOD_DATA_PATH}/version0"
OUTFILEPATH = f"{HOD_DATA_PATH}/HOD_data"

# Ensure reproducibility
RANDOMSTATE = np.random.RandomState(1998)

"""
Ranges we consider for the five parameters h, omega_m, As, ns.
"""



def make_csv_files(
        num_train:  int  = 50, 
        num_test:   int  = 10, 
        num_val:    int  = 10, 
        fix_ng:     bool = True
        ):
    """
    Create parameter datasets.
    N=50 combinations for train,
    N=10 combinations for test and validation
    """

    ### FIDUCIAL VALUES. From https://arxiv.org/pdf/2208.05218.pdf, Table 3  ###
    # log10Mmin   = 13.62 # h^-1 Msun
    sigma_logM  = 0.6915 
    log10M1     = 14.42 # h^-1 Msun
    kappa       = 0.51 
    alpha       = 0.9168  

    ng_desired = 2.174e-4 # h^3 Mpc^-3

    # Vary parameters by 20% around fiducial values 
    param_limits = np.array([
        [sigma_logM * 0.9,  sigma_logM  * 1.1],
        [log10M1    * 0.9,  log10M1     * 1.1],
        [kappa      * 0.9,  kappa       * 1.1],
        [alpha      * 0.9,  alpha       * 1.1],
    ])

    if not fix_ng:
        log10Mmin           = 13.62 # h^-1 Msun
        log10Mmin_limits    = np.array([log10Mmin * 0.9, log10Mmin * 1.1])
        param_limits        = np.vstack((log10Mmin_limits, param_limits))

    dataset_config_size = {
        'train': num_train,
        'test':  num_test,
        'val':   num_val
        }
    dataset_names = ['train', 'test', 'val']

    # Create parameter files. 
    for dataset in dataset_names:

        # Create LHS sampler
        LHC_sampler = LHS(
            xlimits      = param_limits, 
            criterion    = "corr",
            random_state = RANDOMSTATE,
        ) 
        
        # Sample parameters
        samples     = dataset_config_size[dataset]
        node_params = LHC_sampler(samples)
        if fix_ng:
            start       = time.time() 
            log10Mmin   = estimate_log10Mmin_from_gal_num_density(
                sigma_logM_array    = node_params[:, 0],
                log10M1_array       = node_params[:, 1],
                kappa_array         = node_params[:, 2],
                alpha_array         = node_params[:, 3],
                ng_desired          = ng_desired,
                )

            node_params = np.hstack((log10Mmin[:, np.newaxis], node_params))
            print(f"Estimating log10Mmin for {dataset} dataset took {time.time() - start:.2f} seconds.")
        
        # Save parameters to csv file
        df = pd.DataFrame({
            'log10Mmin'     : node_params[:, 0],
            'sigma_logM'    : node_params[:, 1],
            'log10M1'       : node_params[:, 2],
            'kappa'         : node_params[:, 3],
            'alpha'         : node_params[:, 4],
        })
        fname   = Path("HOD_parameters_ng_fixed_" + str(dataset) + ".csv")
        outfile = Path(OUTFILEPATH / fname)
        df.to_csv(
            outfile,
            index=False
        )

make_csv_files(fix_ng=False)

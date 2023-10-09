import numpy as np
import pandas as pd
import time 
from smt.sampling_methods import LHS
from pathlib import Path

from numdens_HOD import estimate_log10Mmin_from_gal_num_density

# DATAPATH = Path("mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation/HOD_parameters")
D13_BASE_PATH       = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files"

# Ensure reproducibility
RANDOMSTATE = np.random.RandomState(1998)

"""
Ranges we consider for the five parameters h, omega_m, As, ns.
"""



def make_csv_files(
        num_train:  int   = 500, 
        num_test:   int   = 100, 
        num_val:    int   = 100, 
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

    ### FIDUCIAL VALUES. From https://arxiv.org/pdf/2208.05218.pdf, Table 3  ###
    # log10Mmin   = 13.62 # h^-1 Msun
    sigma_logM  = 0.6915 
    log10M1     = 14.42 # h^-1 Msun
    kappa       = 0.51 
    alpha       = 0.9168  


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
        version_str = str(version).zfill(3)
        phase_str   = str(phase).zfill(3)
        HOD_PARAMETERS_PATH = Path(f"{D13_BASE_PATH}/AbacusSummit_base_c{version_str}_ph{phase_str}/HOD_parameters")
        HOD_PARAMETERS_PATH.mkdir(parents=True, exist_ok=True)

        fname = f"HOD_parameters_{dataset}"
        if fix_ng:
            fname += "_ng_fixed" 
        fname   = Path(f"{fname}.csv")
        outfile = Path(HOD_PARAMETERS_PATH / fname)
        if outfile.exists():
            print(f"File {outfile} already exists. Skipping.")
            continue

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
        
        df.to_csv(
            outfile,
            index=False
        )

# make_csv_files(num_train=500,
#                num_test=100,
#                num_val=100,
#                fix_ng=True)

def make_csv_files_broad_emulator_grid():
    print("Making HOD parameter files for broad emulator grid, versions 130-181.")
    for ver in range(130,182):
        make_csv_files(
            num_train   = 500,
            num_test    = 100,
            num_val     = 100,
            fix_ng      = True,
            version     = ver,
            phase       = 0,
        )
    print("Done.")

def make_csv_files_linear_derivative_grid():
    print("Making HOD parameter files for linear derivative grid, versions 100-126.")
    for ver in range(100,127):
        make_csv_files(
            num_train   = 500,
            num_test    = 100,
            num_val     = 100,
            fix_ng      = True,
            version     = ver,
            phase       = 0,
        )
    print("Done.")
            
def make_csv_files_c000_all_phases():
    print("Making HOD parameter files for all phases of c000, phases 000-024.")
    for ph in range(0,25):
        make_csv_files(
            num_train   = 500,
            num_test    = 100,
            num_val     = 100,
            fix_ng      = True,
            version     = 0,
            phase       = ph,
        )
    print("Done.")
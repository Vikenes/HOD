import numpy as np
import pandas as pd
import time 
from smt.sampling_methods import LHS
from pathlib import Path
import concurrent.futures

from numdens_HOD import get_log10Mmin_from_varying_log_ng


DATAPATH = Path("mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation/HOD_parameters")
D13_BASE_PATH       = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
D13_EMULATION_PATH  = f"{D13_BASE_PATH}/emulation_files"



def make_csv_files(
        num_train:  int   = 500, 
        num_test:   int   = 100, 
        num_val:    int   = 100, 
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
    sigma_logM  = 0.6915 
    log10M1     = 14.42 # h^-1 Msun
    kappa       = 0.51 
    alpha       = 0.9168  


    # Vary parameters by +/- 10% around fiducial values 
    param_limits = np.array([
        [sigma_logM * 0.9,  sigma_logM  * 1.1],
        [log10M1    * 0.9,  log10M1     * 1.1],
        [kappa      * 0.9,  kappa       * 1.1],
        [alpha      * 0.9,  alpha       * 1.1],
        [-3.7            , -3.2              ],
    ])

    
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
        simname     = f"AbacusSummit_base_c{version_str}_ph{phase_str}"

        # Check if simulation data exists, if not, raise error
        HOD_PARAMETERS_PATH = Path(f"{D13_EMULATION_PATH}/{simname}/HOD_parameters")
        HOD_PARAMETERS_PATH.mkdir(parents=False, exist_ok=True) # Prevent creation of of files for non-existing simulations

        fname   = Path(f"HOD_parameters_{dataset}.csv")
        outfile = Path(HOD_PARAMETERS_PATH / fname)
        if outfile.exists():
            print(f"File {outfile.name} already exists. Skipping.")
            continue

        # Create LHS sampler
        LHC_sampler = LHS(
            xlimits      = param_limits, 
            criterion    = "corr", 
            random_state=0
        ) 
        # Sample parameters
        samples     = dataset_config_size[dataset]
        node_params = LHC_sampler(samples)
        start       = time.time() 
        log10Mmin   = get_log10Mmin_from_varying_log_ng(
            sigma_logM_array    = node_params[:, 0],
            log10M1_array       = node_params[:, 1],
            kappa_array         = node_params[:, 2],
            alpha_array         = node_params[:, 3],
            log_ng_array        = node_params[:, 4],
            version             = version,
            phase               = phase,
            )

        node_params = np.hstack((log10Mmin[:, np.newaxis], node_params))
        print(f"Estimating log10Mmin for {simname}/{dataset} dataset took {time.time() - start:.2f} seconds.")
        # Save parameters to csv file
        df = pd.DataFrame({
            'log10Mmin'     : node_params[:, 0],
            'sigma_logM'    : node_params[:, 1],
            'log10M1'       : node_params[:, 2],
            'kappa'         : node_params[:, 3],
            'alpha'         : node_params[:, 4],
            'log10_ng'      : node_params[:, 5],
        })
        
        # Save to csv file
        df.to_csv(
            outfile,
            index=False
        )
        # else:
            # print("Successfully created parameter file for testing.")
            # print(f" - Parameters for {dataset} dataset has shape {node_params.shape}")


def make_csv_files_phase_wrapper(phase):
    make_csv_files(
        num_train   = 500,
        num_test    = 100,
        num_val     = 100,
        version     = 0,
        phase       = phase,
    )

def make_csv_files_version_wrapper(version):
    make_csv_files(
        num_train   = 500,
        num_test    = 100,
        num_val     = 100,
        version     = version,
        phase       = 0,
    )


def make_csv_files_broad_emulator_grid(parallel=True):
    print("Making HOD parameter files for broad emulator grid, versions 130-181.")
    version_range = range(130,182)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(make_csv_files_version_wrapper, [i for i in version_range])
    print("Done.")

def make_csv_files_linear_derivative_grid(parallel=True, include_c00_=True):
    if include_c00_:
        version_range = np.arange(1,5)
        version_range = np.concatenate((version_range, np.arange(100,127)))
    else:
        version_range = range(100,127)
    print("Making HOD parameter files for linear derivative grid, versions 100-126.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(make_csv_files_version_wrapper, [i for i in version_range])
    print("Done.")
            
def make_csv_files_c000_all_phases(parallel=True):
    phase_range = range(0,25)
    print("Making HOD parameter files for all phases of c000, phases 000-024.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(make_csv_files_phase_wrapper, [i for i in phase_range])

    print("Done.")



# make_csv_files_c000_all_phases(parallel=True)
# make_csv_files_linear_derivative_grid(parallel=True, include_c00_=True)
# make_csv_files_broad_emulator_grid(parallel=True)

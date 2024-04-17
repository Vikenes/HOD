import numpy as np
import pandas as pd
import time 
from smt.sampling_methods import LHS
from pathlib import Path
import concurrent.futures

from numdens_HOD import estimate_log10Mmin_from_gal_num_density

# DATAPATH = Path("mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation/HOD_parameters")
D13_BASE_PATH       = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
D13_EMULATION_PATH  = f"{D13_BASE_PATH}/emulation_files"



def make_csv_files(
        num_train:  int   = 500, 
        num_test:   int   = 100, 
        num_val:    int   = 100, 
        fix_ng:     bool  = True,
        ng_desired: float = 2.174e-4, # h^3 Mpc^-3
        version:    int   = 0,
        phase:      int   = 0,
        test:       bool  = False,
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
    ])

    if not fix_ng:
        # Vary log10Mmin by +/- 10% around fiducial value
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
        simname     = f"AbacusSummit_base_c{version_str}_ph{phase_str}"

        # Check if simulation data exists, if not, raise error
        HOD_PARAMETERS_PATH = Path(f"{D13_EMULATION_PATH}/{simname}/HOD_parameters")
        HOD_PARAMETERS_PATH.mkdir(parents=False, exist_ok=True) # Prevent creation of of files for non-existing simulations

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
            criterion    = "corr"
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
                version             = version,
                phase               = phase,
                test                = False,
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
        })
        
        if not test:
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
        fix_ng      = True,
        version     = 0,
        phase       = phase,
        test        = False
    )

def make_csv_files_version_wrapper(version):
    make_csv_files(
        num_train   = 500,
        num_test    = 100,
        num_val     = 100,
        fix_ng      = True,
        version     = version,
        phase       = 0,
        test        = False
    )

# make_csv_files(num_train=500,
#                num_test=100,
#                num_val=100,
#                fix_ng=True)

def test_make_csv_files(
        version_test=True, 
        phase_test=True, 
        parallel=True
        ):
    if version_test:
        func = make_csv_files_version_wrapper
        arg_range = range(3,6)
        print("Testing make_csv_files for all versions.")
    elif phase_test:
        func = make_csv_files_phase_wrapper
        arg_range = range(0,25)
        print("Testing make_csv_files for all phases.")
    else:
        func = make_csv_files_version_wrapper
        arg_range = range(0,4)
        print("Testing make_csv_files for all versions.")

    if parallel:
        print(f"Using parallel with function {func.__name__} and arg_range {arg_range}")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(func, [i for i in arg_range])
    else:
        make_csv_files(
            num_train   = 100,
            num_test    = 20,
            num_val     = 20,
            fix_ng      = True,
            version     = 0,
            phase       = 0,
            test        = True,
        )



def make_csv_files_broad_emulator_grid(parallel=True):
    print("Making HOD parameter files for broad emulator grid, versions 130-181.")
    version_range = range(130,182)
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(make_csv_files_version_wrapper, [i for i in version_range])
    else:
        for ver in version_range:
            make_csv_files(
                num_train   = 500,
                num_test    = 100,
                num_val     = 100,
                fix_ng      = True,
                version     = ver,
                phase       = 0,
            )
    print("Done.")

def make_csv_files_linear_derivative_grid(parallel=True, include_c00_=True):
    if include_c00_:
        version_range = np.arange(1,5)
        version_range = np.concatenate((version_range, np.arange(100,127)))
    else:
        version_range = range(100,127)
    print("Making HOD parameter files for linear derivative grid, versions 100-126.")
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(make_csv_files_version_wrapper, [i for i in version_range])
    else:
        for ver in version_range:
            make_csv_files(
                num_train   = 500,
                num_test    = 100,
                num_val     = 100,
                fix_ng      = True,
                version     = ver,
                phase       = 0,
            )
    print("Done.")
            
def make_csv_files_c000_all_phases(parallel=True):
    phase_range = range(0,25)
    print("Making HOD parameter files for all phases of c000, phases 000-024.")
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(make_csv_files_phase_wrapper, [i for i in phase_range])
    else:
        for ph in phase_range:
            make_csv_files(
                num_train   = 500,
                num_test    = 100,
                num_val     = 100,
                fix_ng      = True,
                version     = 0,
                phase       = ph,
            )
    print("Done.")



make_csv_files(version=4, phase=0, test=False)

# make_csv_files_linear_derivative_grid(parallel=True, include_c00_=True)
# make_csv_files_broad_emulator_grid(parallel=True)
# make_csv_files_c000_all_phases(parallel=True)

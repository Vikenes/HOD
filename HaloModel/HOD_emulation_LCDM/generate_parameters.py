import numpy as np
import sys
import pandas as pd
from smt.sampling_methods import LHS
from pathlib import Path

# DATAPATH = Path("mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation/HOD_parameters")
DATAPATH = Path("/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation/HOD_data")

# Ensure reproducibility
RANDOMSTATE = np.random.RandomState(1998)

"""
Ranges we consider for the five parameters h, omega_m, As, ns.
"""

### FIDUCIAL VALUES. From https://arxiv.org/pdf/2208.05218.pdf, Table 3  ###
log10Mmin   = 13.62 # h^-1 Msun
sigma_logM  = 0.6915 
log10M1     = 14.42 # h^-1 Msun
kappa       = 0.51 
alpha       = 0.9168  

# Vary parameters by 20% around fiducial values 
param_limits = np.array([
    [log10Mmin  * 0.9,  log10Mmin   * 1.1],
    [sigma_logM * 0.9,  sigma_logM  * 1.1],
    [log10M1    * 0.9,  log10M1     * 1.1],
    [kappa      * 0.9,  kappa       * 1.1],
    [alpha      * 0.9,  alpha       * 1.1],
])

"""
Create parameter datasets.
N=50 combinations for train,
N=10 combinations for test and validation
"""
num_train   = int(50)
num_test    = int(10)
num_val     = int(10)

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
        xlimits=param_limits, 
        criterion="corr",
        random_state=RANDOMSTATE,
    ) 
    
    # Sample parameters
    samples = dataset_config_size[dataset]
    node_params = LHC_sampler(samples)

    # Save parameters to csv file
    df = pd.DataFrame({
        'log10Mmin'     : node_params[:, 0],
        'sigma_logM'    : node_params[:, 1],
        'log10M1'       : node_params[:, 2],
        'kappa'         : node_params[:, 3],
        'alpha'         : node_params[:, 4],
    })
    fname = Path("HOD_parameters_" + str(dataset) + ".csv")
    outfile = Path(DATAPATH / fname)
    df.to_csv(
        outfile,
        index=False
    )

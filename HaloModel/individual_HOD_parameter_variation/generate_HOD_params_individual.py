import numpy as np
import sys
import pandas as pd
from smt.sampling_methods import LHS
from pathlib import Path
from scipy import special
from scipy.integrate import simpson
import matplotlib.pyplot as plt

from numdens_HOD import estimate_log10Mmin_from_gal_num_density

HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
OUTFILEPATH = f"{HOD_DATA_PATH}/individual_HOD_parameter_variation"

# Ensure reproducibility
RANDOMSTATE = np.random.RandomState(1998)

"""
Ranges we consider for the five parameters h, omega_m, As, ns.
"""


### FIDUCIAL VALUES. From https://arxiv.org/pdf/2208.05218.pdf, Table 3  ###
sigma_logM  = 0.6915 
log10M1     = 14.42 # h^-1 Msun
kappa       = 0.51 
alpha       = 0.9168  

ng_desired = 2.174e-4 # h^3 Mpc^-3

param_fiducial = np.array([
    sigma_logM,
    log10M1,
    kappa,
    alpha,
])

# Vary parameters by 20% around fiducial values 
# param_limits = np.array([
#     [sigma_logM * 0.9,  sigma_logM  * 1.1],
#     [log10M1    * 0.9,  log10M1     * 1.1],
#     [kappa      * 0.9,  kappa       * 1.1],
#     [alpha      * 0.9,  alpha       * 1.1],
# ])
param_limits = np.array([
    [sigma_logM ,  sigma_logM  ],
    [log10M1    ,  log10M1     ],
    [kappa      ,  kappa       ],
    [alpha      ,  alpha       ],
])

N_samples = 10 # Number of samples per parameter


dataset_names = ['sigma_logM',
                 'log10M1',
                 'kappa',
                 'alpha']


# Create parameter files. 
for i, dataset in enumerate(dataset_names):
   
    low  = param_limits[i][0] * 0.9 
    high = param_limits[i][1] * 1.1

    lims = param_limits
    lims[i][0] *= 0.9
    lims[i][1] *= 1.1

    LHS_sampler = LHS(
        xlimits=lims, 
        criterion="c",
        random_state=RANDOMSTATE,
    ) 
    
    samples = N_samples #dataset_config_size[dataset]
    node_params = LHS_sampler(samples)

    log10Mmin_best_fit = estimate_log10Mmin_from_gal_num_density(
        sigma_logM_array=node_params[:, 0],
        log10M1_array=node_params[:, 1],
        kappa_array=node_params[:, 2],
        alpha_array=node_params[:, 3],
        ng_desired=ng_desired,
    )


    # Save parameters to csv file
    df = pd.DataFrame({
        'log10Mmin'     : log10Mmin_best_fit,
        'sigma_logM'    : node_params[:, 0],
        'log10M1'       : node_params[:, 1],
        'kappa'         : node_params[:, 2],
        'alpha'         : node_params[:, 3],
    })
    fname = Path("HOD_parameters_vary_" + str(dataset) + ".csv")
    outfile = Path(OUTFILEPATH / fname)
    df.to_csv(
        outfile,
        index=False
    )

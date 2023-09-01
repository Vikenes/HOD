import numpy as np
import sys
import pandas as pd
from smt.sampling_methods import LHS
from pathlib import Path

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


param_fiducial = np.array([
    [sigma_logM ,  sigma_logM  ],
    [log10M1    ,  log10M1     ],
    [kappa      ,  kappa       ],
    [alpha      ,  alpha       ],
])


ng_desired = 2.174e-4 # h^3 Mpc^-3


def generate_parameters(param_limits, subfolder, N_samples=15, overwrite=False):

    dataset_names = ['sigma_logM',
                    'log10M1',
                    'kappa',
                    'alpha']

    # Create parameter files. 
    for i, dataset in enumerate(dataset_names):
        outdir = Path(f"{OUTFILEPATH}/{subfolder}") 
        outdir.mkdir(parents=True, exist_ok=True)
        fname = Path(f"HOD_parameters_vary_{dataset}.csv")
        outfile = Path(outdir / fname)
        if outfile.exists() and not overwrite:
            print(f"File {outfile} already exists, skipping...")
            continue
    
        lims = param_fiducial.copy()


        lims[i] = param_limits[i]

        LHS_sampler = LHS(
            xlimits=lims, 
            criterion="c",
            random_state=RANDOMSTATE,
        ) 
        
        samples = N_samples #dataset_config_size[dataset]
        node_params = np.sort(LHS_sampler(samples), axis=0)

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
        
        df.to_csv(
            outfile,
            index=False
        )



### MIN AND MAX PRIORS from https://arxiv.org/pdf/2208.05218.pdf, Table 3  ###
param_limits_priors = np.array([
    [0.1 , 1.0 ],
    [12.0, 16.0],
    [0.01, 3.0 ],
    [0.5 , 1.0 ],
])

param_limits_ten_percent = np.array([
    [sigma_logM*0.9 ,  sigma_logM*1.1 ],
    [log10M1*0.9    ,  log10M1*1.1    ],
    [kappa*0.9      ,  kappa*1.1      ],
    [alpha*0.9      ,  alpha*1.1      ],
])


if len(sys.argv) >= 11:
    subfolder = sys.argv[1]
    N_samples = int(sys.argv[2])

    overwrite = bool(sys.argv[3]) if len(sys.argv) == 12 else False
    param_lims_start_idx = 4 if len(sys.argv) == 12 else 3
    param_limits = np.array(sys.argv[param_lims_start_idx:], dtype=float).reshape((4, 2))
    generate_parameters(param_limits=param_limits,
                        subfolder=subfolder,
                        N_samples=N_samples,
                        overwrite=overwrite)


elif len(sys.argv) == 3 or len(sys.argv) == 4:

    subfolder = sys.argv[1]
    N_samples = int(sys.argv[2]) #if len(sys.argv) == 3 else 15
    overwrite = bool(sys.argv[3]) if len(sys.argv) == 4 else False
    generate_parameters(param_limits_priors, 
                        subfolder=subfolder, 
                        N_samples=N_samples,
                        overwrite=overwrite)
    
elif len(sys.argv) == 5:
    subfolder = sys.argv[1]
    N_samples = int(sys.argv[2])
    overwrite = bool(sys.argv[3])
    if sys.argv[4] == "ten_percent":
        param_limits = param_limits_ten_percent
    else:
        raise NotImplementedError("Only ten_percent implemented")
    generate_parameters(param_limits=param_limits,
                        subfolder=subfolder,
                        N_samples=N_samples,
                        overwrite=overwrite)

else:
    raise ValueError("Need to specify subfolder")

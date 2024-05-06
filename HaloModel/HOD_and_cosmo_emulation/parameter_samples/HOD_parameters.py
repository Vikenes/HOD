import numpy as np
import sys
import pandas as pd
from pathlib import Path
import h5py 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from HOD_and_cosmo_prior_ranges import get_fiducial_HOD_params

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

params = {'xtick.top': True, 
          'ytick.right': True, 
          'xtick.direction': 'in', 
          'ytick.direction': 'in',
          }
plt.rcParams.update(params)

DATAPATH            = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files")

DATASET_NAMES       = ["train", "test", "val"]
HOD_PARAM_NAMES     = ["alpha", "kappa", "log10M1", "log10Mmin", "sigma_logM", "log10_ng"]


N_SIMS = len([simulation for simulation in DATAPATH.glob("AbacusSummit*")])

def save_HOD_params():
    """
    Store HOD_parameters_{flag}. csv files for each flag  
    """
    for flag in DATASET_NAMES:
        # Loop over flag. Skip file if exists 
        outfile     = Path(f"./HOD_params_{flag}.csv")
        if outfile.exists():
            print(f"{outfile} already exists, skipping")
            continue

        # Empty dataframe to fill with HOD parameters
        HOD_params = pd.DataFrame()
        for simulation in DATAPATH.glob("AbacusSummit*"):
            # Loop over all simulations
            if not Path(simulation / "HOD_parameters" / f"HOD_parameters_{flag}.csv").exists():
                # Skip simulation if it doesn't contain data for this flag
                continue 

            # Concatenate HOD parameters from simulations
            HOD_params = pd.concat([HOD_params, pd.read_csv(simulation / "HOD_parameters" / f"HOD_parameters_{flag}.csv")])
        
        # Save HOD parameters to file
        HOD_params.to_csv(outfile, index=False)

def save_combined_HOD_params():
    # For testing initially. 
    HOD_params_all = pd.DataFrame()
    for flag in DATASET_NAMES:
        HOD_params_all = pd.concat([HOD_params_all, pd.read_csv(f"./HOD_params_{flag}.csv")])
    HOD_params_all.to_csv("HOD_params_all.csv", index=False)

def plot_log10Mmin():
    """
    Make histogram showing the distribution of log10Mmin for each dataset
    """
    fig, ax = plt.subplots()
    flags = {
        "train": "Training",
        "test": "Testing",
        "val": "Validation",
    }
    for flag in DATASET_NAMES:
        HOD_params = pd.read_csv(f"HOD_params_{flag}.csv")

        ax.hist(HOD_params["log10Mmin"], bins=100, histtype="step", label=flags[flag])
    
    ax.legend()
    ax.set_xlabel(r"$\log{M_\mathrm{min}}$")
    ax.set_ylabel("Number of samples")
    if not SAVEFIG:
        plt.show()
        return 
    

    figname_stem = "log10Mmin_hist"
    figname_png = Path(f"plots/thesis_figures_HOD/{figname_stem}.png")
    figname_pdf = Path(f"plots/thesis_figures_HOD/{figname_stem}.pdf")
    print(f"Storing figure {figname_png}")
    fig.savefig(
        figname_png, 
        bbox_inches="tight", 
        dpi=200
        )
    print(f"Storing figure {figname_pdf}")
    fig.savefig(
        figname_pdf, 
        bbox_inches="tight"
        )

global SAVEFIG
SAVEFIG = False

plot_log10Mmin()
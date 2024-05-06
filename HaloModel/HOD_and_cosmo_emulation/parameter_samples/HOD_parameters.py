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
D5_PATH             = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")

DATASET_NAMES       = ["train", "test", "val"]
HOD_PARAM_NAMES     = ["alpha", "kappa", "log10M1", "log10Mmin", "sigma_logM", "log10_ng"]


def save_HOD_params():
    """
    Store HOD_parameters_{flag}. csv files for each flag  
    """
    # Get the simulations used for each dataset

    dataset_simulations = {}
    for flag in DATASET_NAMES:
        fff = h5py.File(D5_PATH / f"TPCF_{flag}.hdf5", "r")
        dataset_simulations[flag] = list(fff.keys())
        fff.close()

    for flag in DATASET_NAMES:
        # Loop over flag. Skip file if exists 
        outfile     = Path(f"./HOD_params_{flag}.csv")
        if outfile.exists():
            print(f"{outfile} already exists, skipping")
            continue

        HOD_params = pd.DataFrame()
        # Loop over all simulations
        for simulation in dataset_simulations[flag]:
            # Concatenate HOD parameters from simulations
            HOD_params = pd.concat([HOD_params, pd.read_csv(DATAPATH / f"{simulation}/HOD_parameters/HOD_parameters_{flag}.csv")])
        
        # Save HOD parameters to file
        # old = pd.read_csv(f"./HOD_params_{flag}.csv")
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
    fig, ax = plt.subplots(figsize=(8, 4))
    # print(fig.get_size_inches(6, 5))
    # exit()
    flags = {
        "train": "Training/5",
        "test": "Testing",
        "val": "Validation",
    }
    weight_factors = {
        "train": 0.2, #1./3.,
        "test":  1.0,
        "val":   1.0,
    }

    colors = {
        "train": "blue",
        "test":  "red",
        "val":   "green",
    }
    for flag in DATASET_NAMES:
        HOD_params = pd.read_csv(f"HOD_params_{flag}.csv")

        ax.hist(
            HOD_params["log10Mmin"], 
            bins=50, 
            histtype="step", 
            weights=np.ones_like(HOD_params["log10Mmin"]) * weight_factors[flag],
            color=colors[flag],
            label=flags[flag],
            )
    
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
# SAVEFIG = True


plot_log10Mmin()
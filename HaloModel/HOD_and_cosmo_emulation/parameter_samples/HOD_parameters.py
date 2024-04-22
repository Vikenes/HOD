import numpy as np
import sys
import pandas as pd
from pathlib import Path
import h5py 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
params = {'xtick.top': True, 
          'ytick.right': True, 
          'xtick.direction': 'in', 
          'ytick.direction': 'in',
          "xtick.labelsize": 8,
          "ytick.labelsize": 8,
          "xtick.major.size": 2,
          "xtick.minor.visible": True,
          "xtick.minor.size": 1,
          "ytick.major.size": 2,
          "ytick.minor.visible": True,
          "ytick.minor.size": 1,
          }
plt.rcParams.update(params)

BASEPATH            = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")
EMULPATH            = Path(BASEPATH / "xi_over_xi_fiducial")
D13_FIDUCIAL_PATH   = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/AbacusSummit_base_c000_ph000/HOD_parameters")

DATASET_NAMES       = ["train", "test", "val"]
HOD_PARAM_NAMES     = ["alpha", "kappa", "log10M1", "log10Mmin", "sigma_logM"]


def save_HOD_params_array():
    for flag in DATASET_NAMES:
        outfile     = Path(f"./HOD_params_{flag}.csv")
        if outfile.exists():
            print(f"{outfile} already exists, skipping")
            continue
        fff         = h5py.File(EMULPATH / f"TPCF_{flag}_ng_fixed.hdf5", "r")
        N_sims      = len([fff[simulation] for simulation in fff.keys() if simulation.startswith("AbacusSummit")])
        for key in fff.keys():
            if not key.startswith("AbacusSummit"):
                continue
            N_nodes = len(fff[key].keys())
            break 
        
        HOD_params  = np.zeros((N_sims, N_nodes, len(HOD_PARAM_NAMES)))
        for ii, simulation in enumerate(fff.keys()):
            if not simulation.startswith("AbacusSummit"):
                continue
            fff_cosmo = fff[simulation]#["node0"] 
            for jj, node in enumerate(fff_cosmo.keys()):
                fff_cosmo_node = fff_cosmo[node]
                HOD_params[ii,jj] = np.array([fff_cosmo_node.attrs[param] for param in HOD_PARAM_NAMES])

        # Reshape HOD_params array to (N_sims*N_nodes, N_params)
        HOD_params = HOD_params.reshape(-1, len(HOD_PARAM_NAMES))

        # Save to csv file
        df = pd.DataFrame({
            key: HOD_params[:, kk] for kk, key in enumerate(HOD_PARAM_NAMES)
        })
        df.to_csv(outfile, index=False)

def load_HOD_params_array():
    HOD_params_dict = {}
    for flag in DATASET_NAMES:
        df = pd.read_csv(f"./HOD_params_{flag}.csv")
        HOD_params_dict[flag] = {
            key: df[key].values for key in HOD_PARAM_NAMES
        }
        hpdf = HOD_params_dict[flag]
    
    df_fiducial = pd.read_csv(D13_FIDUCIAL_PATH / "HOD_parameters_fiducial_ng_fixed.csv")
    HOD_params_dict["fiducial"] = {
        key: df_fiducial[key].values for key in HOD_PARAM_NAMES
    }

    return HOD_params_dict

HOD_params_dict = load_HOD_params_array()


HOD_PARAM_NAMES_LABELS = [r"$\alpha", r"$\kappa$", r"$\log_{10}M_1$", r"$\log_{10}M_{\mathrm{min}}$", r"$\sigma_{\log M}$"]

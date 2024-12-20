import numpy as np
import sys
import pandas as pd
from pathlib import Path
import h5py 
sys.path.append("../")
from make_tpcf_emulation_data_files import train_test_val_paths_split
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
        #   "xtick."
          "xtick.labelsize": 10,
          "ytick.labelsize": 10,
          "xtick.minor.visible": True,
          "xtick.minor.size": 1.5,
          "ytick.major.size": 3,
          "ytick.minor.visible": True,
          "ytick.minor.size": 1.5,
          "xtick.major.size": 3,
          }
plt.rcParams.update(params)

"""
Make a triangle plot of cosmological parameters
Shows samples used for training, testing and validation data sets.
"""



D13_BASE_PATH       = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit")
D13_EMULATION_PATH  = Path(D13_BASE_PATH / "emulation_files")
# COSMOLOGY_PARAM_KEYS    = ["wb", "wc", "Ol", "ln1e10As", "ns", "alpha_s", "w", "w0", "wa", "sigma8", "Om", "h", "N_eff"]
# COSMO_PARAM_NAMES      = ["N_eff", "alpha_s", "ns", "sigma8", "w0", "wa", "wb", "wc"]
COSMO_PARAM_NAMES      = ["wb", "wc", "sigma8", "w0", "wa", "ns", "alpha_s", "N_eff"]

COSMO_PARAM_LABELS = {
            "N_eff"     : r"$N_\mathrm{eff}$",
            "alpha_s"   : r"$\mathrm{d}n_s/\mathrm{d}\ln{k}$",
            "ns"        : r"$n_s$",
            "sigma8"    : r"$\sigma_8$",
            "w0"        : r"$w_0$",
            "wa"        : r"$w_a$",
            "wb"        : r"$\omega_b$",
            "wc"        : r"$\omega_\mathrm{cdm}$",
        }

DATASET_NAMES = ["train", "test", "val"]

cosmo_params_dict = {}


train_paths, test_paths, val_paths = train_test_val_paths_split(split_by_percent=0.8)
# train_paths, test_paths, val_paths = train_test_val_paths_split(split_by_percent=None, seed=1998)

path_dict = {"train": train_paths, "test": test_paths, "val": val_paths}

for flag in DATASET_NAMES:
    cosmo_params = np.zeros((len(path_dict[flag]), len(COSMO_PARAM_NAMES)))
    for ii, path in enumerate(path_dict[flag]):
        # fff = h5py.File(path / f"HOD_catalogues/halocat_{flag}.hdf5", "r")
        cosmo_dict = pd.read_csv(Path(path / "cosmological_parameters.dat"), 
                                     sep=" "
                                     ).iloc[0].to_dict()
        for jj, param in enumerate(COSMO_PARAM_NAMES):
            cosmo_params[ii,jj] = cosmo_dict[param]

    # cosmo_params_dict[flag] 
    cosmo_params_flag_dict = {
        key: cosmo_params[:, kk] for kk, key in enumerate(COSMO_PARAM_NAMES)
    }
    cosmo_params_dict[flag] = cosmo_params_flag_dict
# cosmo_params_dict["fiducial"] = COSMO_PARAMS_FIDUCIAL
cosmo_dict_fidcucial = pd.read_csv(Path(D13_EMULATION_PATH / "AbacusSummit_base_c000_ph000/cosmological_parameters.dat"), 
                                    sep=" "
                                    ).iloc[0].to_dict()
cosmo_params_dict["fiducial"] = {k: cosmo_dict_fidcucial[k] for k in COSMO_PARAM_NAMES}
gridsize = len(COSMO_PARAM_NAMES) 

# Make array of parameter limits as (min, max) for each parameter in cosmo_params_dict
param_limits = np.zeros((len(COSMO_PARAM_NAMES), 2))
for ii, key in enumerate(COSMO_PARAM_NAMES):
    min_param = np.min([np.min(cosmo_params_dict[flag][key]) for flag in DATASET_NAMES])
    max_param = np.max([np.max(cosmo_params_dict[flag][key]) for flag in DATASET_NAMES])

    param_limits[ii, 0] = min_param #- np.abs(min_param) / 10#np.min([np.min(cosmo_params_dict[flag][key]) for flag in DATASET_NAMES])
    param_limits[ii, 1] = max_param #+ np.abs(max_param) / 10#np.max([np.max(cosmo_params_dict[flag][key]) for flag in DATASET_NAMES])


def plot_parameters(savefig=False):
    fig = plt.figure(figsize=(11, 11))

    gs = gridspec.GridSpec(gridsize, gridsize, wspace=0.0, hspace=0.0)
    markers = ["o", "v", "^", "*"]
    colors = ["blue", "green", "red", "darkorange"] 
    
    ms = np.ones(4) * 2.2 
    ms[0] = 1.7
    ms[-1] *= 3.0 
    ms_labels = [5,5,5,7]
    labels = ["Training", "Test", "Validation", "Fiducial"]
    alphas = [0.4, 0.4, 0.4, 1]
    zorders = [1, 2, 3, 100]
    lines = []
    for data_idx, key in enumerate(cosmo_params_dict.keys()):
        df = cosmo_params_dict[key]

        for ii in range(gridsize):
            for jj in range(ii):
                ax = plt.subplot(gs[ii, jj])
                ax.plot(
                    df[COSMO_PARAM_NAMES[jj]],
                    df[COSMO_PARAM_NAMES[ii]],
                    lw=0,
                    marker=markers[data_idx],
                    markersize=ms[data_idx],
                    color=colors[data_idx],
                    zorder=zorders[data_idx],
                    alpha=alphas[data_idx],
                )

                if ii == gridsize-1:
                    ax.set_xlabel(COSMO_PARAM_LABELS[COSMO_PARAM_NAMES[jj]])

                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    # Rotate xtick numbers by 45 degrees
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(45)
                else:
                    ax.set_xticklabels([])
                if jj == 0:
                    # ax.set_ylabel(COSMO_PARAM_NAMES_LABELS[ii])
                    ax.set_ylabel(COSMO_PARAM_LABELS[COSMO_PARAM_NAMES[ii]])



                    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                else:
                    ax.set_yticklabels([])
                # if set_lim:
                #     ax.set_xlim(param_limits[jj])
                #     ax.set_ylim(param_limits[ii])

        line, = plt.plot([], [], lw=0, marker=markers[data_idx], markersize=ms_labels[data_idx], color=colors[data_idx], label=labels[data_idx])
        lines.append(line)

    fig.legend(handles=lines, bbox_to_anchor=(0.8, 0.8), loc="upper right", ncol=1, frameon=False) 


    if not savefig:
        fig.tight_layout()
        plt.show()
    else:
        figname_stem = "cosmo_params_train_test_val_fiducial"
        outfig_png = Path(f"plots/thesis_figures_cosmo/{figname_stem}.png")
        outfig_pdf = Path(f"plots/thesis_figures_cosmo/{figname_stem}.pdf")

        print(f"Storing figure {outfig_png}")
        plt.savefig(
            outfig_png,
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=200,
        )
        
        print(f"Storing figure {outfig_pdf}")
        plt.savefig(
            outfig_pdf,
            bbox_inches="tight",
            pad_inches=0.05,
        )

plot_parameters(savefig=False)
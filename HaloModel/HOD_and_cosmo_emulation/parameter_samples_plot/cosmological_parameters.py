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

D13_BASE_PATH       = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit")
D13_EMULATION_PATH  = Path(D13_BASE_PATH / "emulation_files")
# COSMOLOGY_PARAM_KEYS    = ["wb", "wc", "Ol", "ln1e10As", "ns", "alpha_s", "w", "w0", "wa", "sigma8", "Om", "h", "N_eff"]
COSMO_PARAM_NAMES      = ["N_eff", "alpha_s", "ns", "sigma8", "w0", "wa", "wb", "wc"]
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
COSMO_PARAM_NAMES_LABELS = [COSMO_PARAM_LABELS[key] for key in COSMO_PARAM_NAMES]

# Make array of parameter limits as (min, max) for each parameter in cosmo_params_dict
param_limits = np.zeros((len(COSMO_PARAM_NAMES), 2))
for ii, key in enumerate(COSMO_PARAM_NAMES):
    min_param = np.min([np.min(cosmo_params_dict[flag][key]) for flag in DATASET_NAMES])
    max_param = np.max([np.max(cosmo_params_dict[flag][key]) for flag in DATASET_NAMES])
    # print(f"{min_param=}")
    # print(f"{max_param=}")
    # exit()

    param_limits[ii, 0] = min_param #- np.abs(min_param) / 10#np.min([np.min(cosmo_params_dict[flag][key]) for flag in DATASET_NAMES])
    param_limits[ii, 1] = max_param #+ np.abs(max_param) / 10#np.max([np.max(cosmo_params_dict[flag][key]) for flag in DATASET_NAMES])

fig = plt.figure(figsize=(11, 11))

gs = gridspec.GridSpec(gridsize, gridsize, wspace=0.0, hspace=0.0)
markers = ["o", "v", "^", "*"]
colors = ["blue", "green", "red", "gold"] 
    
ms = np.ones(4) * 3 
ms[0] = 2
ms[-1] *= 2 
labels = ["Training", "Test", "Validation", "Fiducial"]
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
            )

            if ii == gridsize-1:
                ax.set_xlabel(COSMO_PARAM_NAMES_LABELS[jj])

                ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            else:
                ax.set_xticklabels([])
            if jj == 0:
                ax.set_ylabel(COSMO_PARAM_NAMES_LABELS[ii])


                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            else:
                ax.set_yticklabels([])
            # if set_lim:
            #     ax.set_xlim(param_limits[jj])
            #     ax.set_ylim(param_limits[ii])

    line, = plt.plot([], [], lw=0, marker=markers[data_idx], markersize=2*ms[data_idx], color=colors[data_idx], label=labels[data_idx])
    lines.append(line)
fig.legend(handles=lines, bbox_to_anchor=(0.8, 0.8), loc="upper right", ncol=1, frameon=False) 


fig.tight_layout()

# plt.show()

plt.savefig(
    f"./plots/cosmo_params.png",
    bbox_inches="tight",
    pad_inches=0.05,
    dpi=200,
)

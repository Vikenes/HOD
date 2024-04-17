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

BASEPATH = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")
EMULPATH = Path(BASEPATH / "xi_over_xi_fiducial")
DATASET_NAMES = ["train", "test", "val"]

HOD_PARAM_NAMES = ["log10Mmin", "sigma_logM", "kappa", "alpha", "log10M1"]
with h5py.File(EMULPATH / f"TPCF_test_ng_fixed.hdf5", "r") as f:
    ff = f["AbacusSummit_base_c000_ph000"]["node0"]
    COSMO_PARAM_NAMES = [k for k in ff.attrs.keys() if k not in HOD_PARAM_NAMES]
    COSMO_PARAMS_FIDUCIAL = {k: ff.attrs[k] for k in COSMO_PARAM_NAMES}

cosmo_params_dict = {}

for flag in DATASET_NAMES:
    fff = h5py.File(EMULPATH / f"TPCF_{flag}_ng_fixed.hdf5", "r")
    N_sims = len([fff[simulation] for simulation in fff.keys() if simulation.startswith("AbacusSummit")])
    cosmo_params = np.zeros((N_sims, len(COSMO_PARAM_NAMES)))

    for ii, simulation in enumerate(fff.keys()):
        if not simulation.startswith("AbacusSummit"):
            continue
        fff_cosmo_node = fff[simulation]["node0"] 
        for jj, param in enumerate(COSMO_PARAM_NAMES):
            cosmo_params[ii,jj] = fff_cosmo_node.attrs[param]

    # cosmo_params_dict[flag] 
    cosmo_params_flag_dict = {
        key: cosmo_params[:, kk] for kk, key in enumerate(COSMO_PARAM_NAMES)
    }
    cosmo_params_dict[flag] = cosmo_params_flag_dict

cosmo_params_dict["fiducial"] = COSMO_PARAMS_FIDUCIAL
# print(COSMO_PARAM_NAMES)
# exit()
gridsize = len(COSMO_PARAM_NAMES) 
COSMO_PARAM_NAMES_LABELS = []
for name in COSMO_PARAM_NAMES:
    if len(name) == 2:
        if name[0] == "w":
            gg = rf"\omega_{name[1]}"
        else:
            gg = f"{name[0]}_{name[1]}"

    elif name=="N_eff":
        g1, g2 = name.split("_")
        gg = rf"{g1}_\mathrm{{{g2}}}"
    elif name=="alpha_s":
        gg = rf"\{name}"

    
    elif name[-1] == "8":
        gg = rf"\{name[:-1]}_{name[-1]}"

    COSMO_PARAM_NAMES_LABELS.append(rf"$\displaystyle {gg}$")

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

    line, = plt.plot([], [], lw=0, marker=markers[data_idx], markersize=2*ms[data_idx], color=colors[data_idx], label=labels[data_idx])
    lines.append(line)
fig.legend(handles=lines, bbox_to_anchor=(0.8, 0.8), loc="upper right", ncol=1, frameon=False) 


fig.tight_layout()

plt.show()

# plt.savefig(
#     f"./plots/cosmo_params.png",
#     bbox_inches="tight",
#     pad_inches=0.05,
#     dpi=200,
# )

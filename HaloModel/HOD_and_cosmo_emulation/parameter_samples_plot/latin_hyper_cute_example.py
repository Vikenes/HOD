import numpy as np
import sys
import pandas as pd
from pathlib import Path
import h5py 

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

BASEPATH = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")
EMULPATH = Path(BASEPATH / "xi_over_xi_fiducial")
DATASET_NAMES = ["train", "test", "val"]

HOD_PARAM_NAMES = ["log10Mmin", "sigma_logM", "kappa", "alpha", "log10M1"]
with h5py.File(EMULPATH / f"TPCF_train_ng_fixed.hdf5", "r") as f:
    ff = f["AbacusSummit_base_c130_ph000"]["node0"]
    COSMO_PARAM_NAMES = [k for k in ff.attrs.keys() if k not in HOD_PARAM_NAMES]


# f = h5py.File(EMULPATH / f"TPCF_train_ng_fixed.hdf5", "r")
# ff = f["AbacusSummit_base_c130_ph000"]["node0"]
# COSMO_PARAMS = [k for k in ff.attrs.keys() if k not in HOD_PARAMS]
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
    break
# print(COSMO_PARAM_NAMES)
# exit()
df = cosmo_params_dict["train"]
N_params = len(COSMO_PARAM_NAMES)
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(N_params-1, N_params-1)
grid = np.zeros((N_params, N_params), dtype=object)
for ii, param_ii in enumerate(df):
    for jj, param_jj in enumerate(df):
        pp = f"{param_ii}+{param_jj}"
        # grid[ii,jj] = pp


# for i in range(N_params,0,-1):
#     for j in range(N_params,0,-1):
#         print(f"{grid[i-1,j-1]:15}", end=", ")
    # print("|")

# for i in range(N_params):
#     for j in range(N_params):
#         print(f"{grid[i,j]:15}", end=",")
#     print("|")

# exit()



num_samples = int(100)
Om0_min = 0.2
Om0_max = 0.6

h_min = 0.5
h_max = 0.9

sigma8_min = 0.6
sigma8_max = 1.0


param_limits = np.array([
    [Om0_min, Om0_max],
    [sigma8_min, sigma8_max],
    [h_min, h_max],
])
from smt.sampling_methods import LHS
LHC_sampler = LHS(xlimits=param_limits, criterion="corr") # https://smt.readthedocs.io/en/latest/_src_docs/sampling_methods/lhs.html
node_params = LHC_sampler(num_samples)
df = pd.DataFrame({
    'Om0'   : node_params[:, 0],
    'sigma8': node_params[:, 1],
    'h'     : node_params[:, 2],
})
gs = gridspec.GridSpec(2, 2)
ax00 = plt.subplot(gs[0, 0])
ax11 = plt.subplot(gs[1, 1])
ax10 = plt.subplot(gs[1, 0])
ax10.plot(
    df['Om0'],
    df['sigma8'],
    lw=0,
    marker='o',
    markersize=1,
    color='k',
)
ax00.plot(
    df['Om0'],
    df['h'],
    lw=0,
    marker='o',
    markersize=1,
    color='k',
)
ax11.plot(
    df['h'],
    df['sigma8'],
    lw=0,
    marker='o',
    markersize=1,
    color='k',
)

ax10.set_xlabel(r'$\Omega_{m0}$')
ax11.set_xlabel(r'$h$')
ax00.set_ylabel(r'$h$')
ax10.set_ylabel(r'$\sigma_8$')

# plt.show()
# exit()
plt.savefig(
    f"./plots/params_LHS_TEST.png",
    bbox_inches="tight",
    pad_inches=0.05,
    dpi=200,
)

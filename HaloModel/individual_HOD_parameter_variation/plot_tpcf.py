import numpy as np 
import sys 
from pathlib import Path
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'axes.labelsize': 20})
matplotlib.rcParams.update({'legend.fontsize': 12})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)


HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
FIGPATH = "plots"

global SAVE
global NG_FIXED
SAVE = False
NG_FIXED = False


def plot_TPCF_individual_param_varied(subfolder):
    """
    Halo_file: halocatalogue file for training parameters 
     - halo_file.keys(): individual files, ['node0', 'node1', ..., 'nodeN']
     - halo_file.attrs.keys(): cosmological parameters, e.g. ['H0', 'Om0', 'lnAs', 'n_s', ...]
     - halo_file['nodex'].attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
     - halo_file['nodex'].keys(): catalogue data, e.g. ['host_radius', 'x', 'y', 'z', 'v_x', ...]
    """

    PARAM_PATH = f"{HOD_DATA_PATH}/individual_HOD_parameter_variation/{subfolder}"
    INDATAPATH = Path(f"{PARAM_PATH}/tpcf_data")
    FIGPATH = f"plots"
    Path(FIGPATH).mkdir(parents=True, exist_ok=True)


    fig = plt.figure(figsize=(8, 6))

    gs = gridspec.GridSpec(
        2, 2,
        hspace=0.0,#/6.0,
        wspace=0
    )
    fig.tight_layout()


    ax0 = plt.subplot(gs[0, 0], )
    ax1 = plt.subplot(gs[0, 1], sharey=ax0)
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1], sharey=ax2)

    

    axes = [ax0, ax1, ax2, ax3]


    params_varied = ['sigma_logM', 'log10M1', 'kappa', 'alpha']
    params_varied_lables = [
        r"$\sigma_{\log M}$", 
        r"$\log M_1$", 
        r"$\kappa$", 
        r"$\alpha$"]
    
    r_fiducial, xi_fiducial = np.load(f"{HOD_DATA_PATH}/individual_HOD_parameter_variation/data_results/TPCF_fiducial.npy")
    
    
    for i, vary_param in enumerate(params_varied):
        param_file = f"{PARAM_PATH}/HOD_parameters_vary_{vary_param}.csv"
        param_csv = pd.read_csv(param_file)
        param_col = param_csv[f'{vary_param}']

        ax = axes[i]
        sorted_idx = np.argsort(param_col)
        param_max_idx = np.argmax(param_col)
        param_min_idx = np.argmin(param_col)


        ax.plot(r_fiducial, xi_fiducial, 
                '-', lw=1, color='k', alpha=1, 
                label="Fiducial", zorder=0)
        

        for j in sorted_idx:#, npy_file in enumerate(INDATAPATH.glob(npy_files)):

            r, xi = np.load(f"{INDATAPATH}/TPCF_vary_{vary_param}_node{j}.npy")

            if j == param_min_idx:
                ax.plot(r, xi, 
                        'o-', ms=2,
                        lw=0.7, alpha=1,
                        color='red', 
                        label=f"{params_varied_lables[i]}={param_col[j]:.3f}",
                        zorder=1)
            elif j == param_max_idx:
                ax.plot(r, xi, 
                        'o-', ms=2,
                        lw=0.7, alpha=1, 
                        color='blue',
                        label=f"{params_varied_lables[i]}={param_col[j]:.3f}",
                        zorder=2)
            else:
                ax.plot(r, xi, '--', lw=0.7, alpha=0.7)


        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='lower left')



    fig.supxlabel(r"$r\: [h^{-1} \mathrm{Mpc}]$")
    fig.supylabel(r"$\xi_{gg}(r)$")
    fig.suptitle(r"Varying individual HOD parameters")

    ax1.tick_params(axis='y', labelleft=False)
    ax3.tick_params(axis='y', labelleft=False)

    ax0.set_ylim(1e-3, 1e6)
    ax2.set_ylim(1e-3, 1e6)

    # plt.show()
    plt.savefig(f"{FIGPATH}/TPCF_{subfolder}.png", dpi=300)
            




SAVE = False
NG_FIXED = True

if len(sys.argv) == 2:
    subfolder = sys.argv[1]
    plot_TPCF_individual_param_varied(subfolder=subfolder)

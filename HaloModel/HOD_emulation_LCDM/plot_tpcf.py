import numpy as np 
import h5py 
import time 
from pathlib import Path
from pycorr import TwoPointCorrelationFunction
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'axes.labelsize': 20})
matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
INDATAPATH = Path(f"{HOD_DATA_PATH}/data_measurements")
FIGPATH = "plots"

global SAVE
global NG_FIXED
SAVE = False
NG_FIXED = False

def compute_TPCF_average(n_bins=128):
    
    npy_files = f"TPCF_train_node*_{n_bins}bins*.npy"

    xi_bar = np.mean([np.load(npy_file)[1] for npy_file in INDATAPATH.glob(npy_files)], axis=0)
    return xi_bar


def plot_TPCF_train_halocats_interval(interval=5, n_bins=128, mask=False, flag="train"):
    """
    Halo_file: halocatalogue file for training parameters 
     - halo_file.keys(): individual files, ['node0', 'node1', ..., 'nodeN']
     - halo_file.attrs.keys(): cosmological parameters, e.g. ['H0', 'Om0', 'lnAs', 'n_s', ...]
     - halo_file['nodex'].attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
     - halo_file['nodex'].keys(): catalogue data, e.g. ['host_radius', 'x', 'y', 'z', 'v_x', ...]
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    xi_bar = compute_TPCF_average(n_bins=n_bins)


    npy_files = f"TPCF_{flag}_node*_{n_bins}bins*.npy"

    for i, npy_file in enumerate(INDATAPATH.glob(npy_files)):

        r, xi = np.load(npy_file)
        if mask:
            r_mask_low_limit = r > 0.6 
            r_mask_high_limit = r < 60
            r_mask = r_mask_low_limit * r_mask_high_limit

            r = r[r_mask]
            xi = xi[r_mask]
            xi_bar = xi_bar[r_mask] if i == 0 else xi_bar

        if interval>10:
            ax.plot(r, xi, 'o-', ms=2, lw=0.7, alpha=0.6)
        else:
            ax.plot(r, xi, 'o-', alpha=0.6, label=f"node{i}")

        if (i+1) % interval == 0 or i == 49:
            node_range = f"{i-interval+1}-{i}"
            fig_title = rf"TPCF, subsamples ${node_range}$"
            if NG_FIXED:
                fig_title += r" with fixed $\bar{n}_g$"
            ax.set_title(fig_title)

            
            ax.plot(r, xi_bar, color='k', label=rf"$\bar{{\xi}}_{{gg}}(r)$")
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r"$\xi_{gg}(r)$")
            ax.set_xlabel(r"$r\: [h^{-1} \mathrm{Mpc}]$")
            ax.set_ylim(1e-3, 1e6)
            plt.legend()
            if SAVE:
                figname = f"TPCF_{flag}_node{node_range}_{n_bins}bins"
                if NG_FIXED:
                    figname += "_ng_fixed"
                plt.savefig(f"{FIGPATH}/{figname}.png", dpi=300)
            else:
                plt.show()
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))     

    plt.close()       
            

def plot_xi_over_xi_bar(interval=5, n_bins=128, mask=False, flag="train"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    xi_bar = compute_TPCF_average(n_bins=n_bins)

    npy_files = f"TPCF_{flag}_node*_{n_bins}bins*.npy"

    for i, npy_file in enumerate(INDATAPATH.glob(npy_files)):
        r, xi = np.load(npy_file)
        if mask:
            r_mask_low_limit = r > 0.6 
            r_mask_high_limit = r < 60
            r_mask = r_mask_low_limit * r_mask_high_limit

            r = r[r_mask]
            xi = xi[r_mask]
            xi_bar = xi_bar[r_mask] if i == 0 else xi_bar
        if interval>10:
            ax.plot(r, xi/xi_bar, '-o', ms=2, alpha=0.6)
        else:
            ax.plot(r, xi/xi_bar, '--', alpha=0.6, label=f"node{i}")

        if (i+1) % interval == 0 or i == 49:
            node_range = f"{i-interval+1}-{i}"

            fig_title = f"TPCF-ratio, subsamples ${node_range}$"
            if NG_FIXED:
                fig_title += r" with fixed $\bar{n}_g$"
            ax.set_title(fig_title)

            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r"$\xi_{gg}(r)/\bar{\xi}_{gg}(r)$")
            ax.set_xlabel(r"$r\: [h^{-1} \mathrm{Mpc}]$")
            if interval <= 10:
                plt.legend()
            if SAVE:
                figname = f"TPCF_{flag}_node{node_range}_{n_bins}bins_ratio"
                if NG_FIXED:
                    figname += "_ng_fixed"

                plt.savefig(f"{FIGPATH}/{figname}.png", dpi=300)
            else:
                plt.show()
            fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 



    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # plt.show()
    # xi_all = np.array(xi_all)
    # xi_mean = np.mean(xi_all, axis=0)
    # ax.plot(r, xi_mean, color='k', label=r"$\bar{\xi}(r)$")
    plt.close()       




SAVE = False
NG_FIXED = True
plot_TPCF_train_halocats_interval(interval=10, n_bins=115, mask=True, flag='test')
# plot_xi_over_xi_bar(interval=25, n_bins=115, mask=True)
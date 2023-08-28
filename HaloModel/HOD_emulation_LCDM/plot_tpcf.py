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
    # xi_all = []
    # for i, npy_file in enumerate(INDATAPATH.glob(f"TPCF_train_node*_{n_bins}bins.npy")):
        # xi_all.append(np.load(npy_file)[1])

    if NG_FIXED:
        npy_files = f"TPCF_train_node*_{n_bins}bins_ng_fixed.npy"
    else:
        npy_files = f"TPCF_train_node*_{n_bins}bins.npy"

    xi_bar = np.mean([np.load(npy_file)[1] for npy_file in INDATAPATH.glob(npy_files)], axis=0)
    return xi_bar


def plot_TPCF_train_halocats_interval(interval=5, n_bins=128):
    """
    Halo_file: halocatalogue file for training parameters 
     - halo_file.keys(): individual files, ['node0', 'node1', ..., 'nodeN']
     - halo_file.attrs.keys(): cosmological parameters, e.g. ['H0', 'Om0', 'lnAs', 'n_s', ...]
     - halo_file['nodex'].attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
     - halo_file['nodex'].keys(): catalogue data, e.g. ['host_radius', 'x', 'y', 'z', 'v_x', ...]
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # xi_all = []
    if NG_FIXED:
        npy_files = f"TPCF_train_node*_{n_bins}bins_ng_fixed.npy"
    else:
        npy_files = f"TPCF_train_node*_{n_bins}bins.npy"
    for i, npy_file in enumerate(INDATAPATH.glob(npy_files)):
        # tpcf_filename = f"{INDATAPATH}/TPCF_train_node{node_idx}.npy"

        r, xi = np.load(npy_file)
        # xi_all.append(xi)
        if interval>10:
            ax.plot(r, xi, '--', alpha=0.6)
        else:
            ax.plot(r, xi, '--', alpha=0.6, label=f"node{i}")

        if (i+1) % interval == 0 or i == 49:
            node_range = f"{i-interval+1}-{i}"
            fig_title = rf"TPCF computed from {n_bins} values of $r\in[0.1,\,130]$, for samples ${node_range}$."
            if NG_FIXED:
                fig_title += r" ($\bar{n}_g$ fixed)"
            ax.set_title(fig_title)
            # xi_mean = np.mean(xi_all, axis=0)
            xi_bar = compute_TPCF_average(n_bins=n_bins)
            ax.plot(r, xi_bar, color='k', label=rf"$\bar{{\xi}}(r)$")
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r"$\xi(r)$")
            ax.set_xlabel(r"$r\: [h^{-1} \mathrm{Mpc}]$")
            plt.legend()
            if SAVE:
                figname = f"TPCF_train_node{node_range}_{n_bins}bins"
                if NG_FIXED:
                    figname += "_ng_fixed"
                plt.savefig(f"{FIGPATH}/{figname}.png", dpi=300)
            else:
                plt.show()
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))     

    plt.close()       
            

def plot_xi_over_xi_bar(interval=5, n_bins=128):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    xi_bar = compute_TPCF_average(n_bins=n_bins)
    if NG_FIXED:
        npy_files = f"TPCF_train_node*_{n_bins}bins_ng_fixed.npy"
    else:
        npy_files = f"TPCF_train_node*_{n_bins}bins.npy"

    for i, npy_file in enumerate(INDATAPATH.glob(npy_files)):
        r, xi = np.load(npy_file)
        if interval>10:
            ax.plot(r, xi/xi_bar, '--', alpha=0.6)
        else:
            ax.plot(r, xi/xi_bar, '--', alpha=0.6, label=f"node{i}")

        if (i+1) % interval == 0 or i == 49:
            node_range = f"{i-interval+1}-{i}"

            # xi_mean = np.mean(xi_all, axis=0)
            # ax.plot(r, xi_mean, color='k', label=r"$\bar{\xi}(r)$")
            # xi_all = []
            fig_title = f"TPCF ratio from {n_bins} values of $r\in[0.1,\,130]$, for samples ${node_range}$"
            if NG_FIXED:
                fig_title += r" ($\bar{n}_g$ fixed)"
            ax.set_title(fig_title)

            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(r"$\xi(r)/\bar{\xi}(r)$")
            ax.set_xlabel(r"$r\: [h^{-1} \mathrm{Mpc}]$")
            if interval <= 10:
                plt.legend()
            if SAVE:
                figname = f"TPCF_train_node{node_range}_{n_bins}bins_ratio"
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

   



SAVE = True
NG_FIXED = True
plot_TPCF_train_halocats_interval(interval=25, n_bins=64)
plot_xi_over_xi_bar(interval=25, n_bins=64)
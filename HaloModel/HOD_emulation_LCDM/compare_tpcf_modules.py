import numpy as np 
import h5py 
import time 
from pathlib import Path
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'axes.labelsize': 20})
matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

# from pycorr import TwoPointCorrelationFunction
# from halotools.mock_observables import tpcf

from compute_tpcf import compute_TPCF_fiducial_halocat_halotools, compute_TPCF_fiducial_halocat


FIGPATH = "plots"
global SAVE
SAVE = True 

def compare_TPCF_times(iterations=10, threads=12, package='pycorr'):
    times = np.empty(iterations)
    if package == 'pycorr':
        for i in range(iterations):
            _, __, duration = compute_TPCF_fiducial_halocat(n_bins=128, threads=threads)
            times[i] = duration

    elif package == 'halotools':
        for i in range(iterations):
            _, __, duration = compute_TPCF_fiducial_halocat_halotools(n_bins=128, threads=threads)
            times[i] = duration

    print(f"Timing results for TPCF computed with {package}")
    print(f"Performing {iterations} iterations, with {threads} threads:")
    print(f" - Average time: {np.mean(times):.2f} s")
    print(f" - Best time   : {np.min(times):.2f} s")
    print(f" - Std dev     : {np.std(times):.2f} s")



def compare_TPCF_fiducial_plot(n_bins=128, threads=12):
    r_halotools, xi_halotools, duration_halotools = compute_TPCF_fiducial_halocat_halotools(n_bins=n_bins, threads=threads)
    r_pycorr, xi_pycorr, duration_pycorr          = compute_TPCF_fiducial_halocat(n_bins=n_bins, threads=threads)
    # r_halotools, xi_halotools, duration_halotools = compute_TPCF_fiducial_halocat(n_bins=n_bins, threads=threads)
    # r_halotools, xi_halotools = r_pycorr, xi_pycorr - 1e-4



    fig = plt.figure(figsize=(8.0, 8.0))

    gs = gridspec.GridSpec(
        2, 1,
        hspace=0,
        height_ratios=[2, 1]
    )

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.set_xscale('log')
    ax0.xaxis.set_ticklabels([])
    ax1.set_xscale('log')
    ax0.set_yscale('log')
    ax1.set_yscale('log')


    ax0.plot(
        r_pycorr, 
        xi_pycorr, 
        's-',
        lw=2,
        ms=4,
        color='blue', 
        label='pycorr'
        )
    ax0.plot(
        r_halotools, 
        xi_halotools, 
        # 'x-',
        ls='dashed', 
        marker='|',
        lw=1,
        ms=16,
        color='red', 
        # alpha=0.7, 
        label='halotools'
        )

   
    ax1.plot(
        r_pycorr, 
        np.abs(xi_pycorr/xi_halotools - 1), 
        color='k', 
        )
    
    fig.suptitle(rf"$\mathrm{{Comparing\: TPCF\: packages, using\:}} {n_bins} \mathrm{{\:bins\: of\:}} r\in[0.1,\,130]$")


    ax0.set_ylabel(r"$\xi_{gg}(r)$")
    ax1.set_ylabel(r"$|\xi_{\rm pycorr}/\xi_{\rm halotools} - 1|$")
    ax1.set_xlabel(r"$r\: [h^{-1} \mathrm{Mpc}]$")

    ax0.legend()
    fig.tight_layout()
    if SAVE:
        plt.savefig(f"{FIGPATH}/compare_TPCF_{n_bins}bins_halotools_pycorr.png", dpi=300)
    else:
        plt.show()

# compare_TPCF_times(iterations=10, threads=128, package='pycorr')
compare_TPCF_fiducial_plot(n_bins=64, threads=128)
import numpy as np 
import h5py 
import time 
from pathlib import Path
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

# from pycorr import TwoPointCorrelationFunction
# from halotools.mock_observables import tpcf

from compute_tpcf import compute_TPCF_fiducial_halocat_halotools, compute_TPCF_fiducial_halocat

HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
HALO_ARRAYS_PATH = f"{HOD_DATA_PATH}/version0"

INDATAPATH = f"{HOD_DATA_PATH}/HOD_data"
OUTDATAPATH = f"{HOD_DATA_PATH}/data_measurements"
dataset_names = ['train', 'test', 'val']

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



def compare_TPCF_fiducial_halocat(n_bins=128):
    r_pycorr, xi_pycorr         = compute_TPCF_fiducial_halocat(n_bins=n_bins)
    r_halotools, xi_halotools   = compute_TPCF_fiducial_halocat_halotools(n_bins=n_bins)

    plt.title(rf"Comparing TPCF packages, using {n_bins} values of $r\in[0.1,\,130]$")
    plt.plot(r_pycorr, xi_pycorr, label='pycorr')
    plt.plot(r_halotools, xi_halotools, '--', label='halotools')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

# read_hdf5_files()
# read_csv_original()
# compute_TPCF_fiducial_halocat(n_bins=64)
# compute_TPCF_train_halocats_halotools(n_bins=64)
# compute_TPCF_fiducial_halocat_halotools(n_bins=64)
# compare_TPCF_fiducial_halocat(n_bins=64)
# compare_TPCF_times(iterations=10, threads=128, package='pycorr')

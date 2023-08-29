import numpy as np 
import h5py 
import time 
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from pycorr import TwoPointCorrelationFunction
from halotools.mock_observables import tpcf

HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
HALO_ARRAYS_PATH = f"{HOD_DATA_PATH}/version0"

INDATAPATH = f"{HOD_DATA_PATH}/HOD_data"
OUTDATAPATH = f"{HOD_DATA_PATH}/data_measurements"
dataset_names = ['train', 'test', 'val']


def compute_TPCF_fiducial_halocat(n_bins=128, threads=12):
    """
    fff.keys(): catalogue data, e.g. ['host_mass', 'host_id', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z', ...]
    fff.attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
    
    """
    fff = h5py.File(f"{INDATAPATH}/halocat_fiducial.hdf5", "r")


    x = np.array(fff['x'][:])
    y = np.array(fff['y'][:])
    z = np.array(fff['z'][:])

    r_bin_edges = np.logspace(np.log10(0.1), np.log10(130), n_bins)
    t0 = time.time()
    result = TwoPointCorrelationFunction(
        mode='s',
        edges=r_bin_edges,
        data_positions1=np.array([x, y, z]),
        boxsize=2000.0,
        engine='corrfunc',
        nthreads=threads,
    )

    r, xi = result(
        return_sep=True
    )
    duration = time.time() - t0
    
    print(f"Time elapsed: {duration:.2f} s")

    return r, xi, duration

def compute_TPCF_fiducial_halocat_halotools(n_bins=128, threads=12):
    """
    fff.keys(): catalogue data, e.g. ['host_mass', 'host_id', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z', ...]
    fff.attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
    
    """
    fff = h5py.File(f"{INDATAPATH}/halocat_fiducial.hdf5", "r")

    x = np.array(fff['x'][:])
    y = np.array(fff['y'][:])
    z = np.array(fff['z'][:])

    r_bin_edges = np.logspace(np.log10(0.1), np.log10(130), n_bins)

    t0 = time.time()
    xi = tpcf(
        sample1=np.array([x, y, z]).T, # halotools requires shape (Npts, 3)
        rbins=r_bin_edges,
        period=2000.0,
        num_threads=threads,
    )

    
    duration = time.time() - t0
    print(f"Time elapsed: {duration:.2f} s")

    r_bin_centers = 0.5*(r_bin_edges[1:] + r_bin_edges[:-1])

    return r_bin_centers, xi, duration



def compute_TPCF_train_halocats_pycorr(n_bins=128, ng_fixed=False):
    """
    Halo_file: halocatalogue file for training parameters 
     - halo_file.keys(): individual files, ['node0', 'node1', ..., 'nodeN']
     - halo_file.attrs.keys(): cosmological parameters, e.g. ['H0', 'Om0', 'lnAs', 'n_s', ...]
     - halo_file['nodex'].attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
     - halo_file['nodex'].keys(): catalogue data, e.g. ['host_radius', 'x', 'y', 'z', 'v_x', ...]
    """

    halocat_fname = f"halocat_train"
    if ng_fixed:
        halocat_fname += "_ng_fixed"

    halo_file = h5py.File(f"{INDATAPATH}/{halocat_fname}.hdf5", "r")
    N_nodes = len(halo_file.keys())
    r_bin_edges = np.logspace(np.log10(0.1), np.log10(130), n_bins)
    
    print(f"Computing TPCF for {N_nodes} nodes...")
    t0 = time.time()
    
    for node_idx in range(N_nodes):
        node_catalogue = halo_file[f"node{node_idx}"]
        x = np.array(node_catalogue['x'][:])
        y = np.array(node_catalogue['y'][:])
        z = np.array(node_catalogue['z'][:])

        t0i = time.time()

        result = TwoPointCorrelationFunction(
            mode='s',
            edges=r_bin_edges,
            data_positions1=np.array([x, y, z]),
            boxsize=2000.0,
            engine='corrfunc',
            nthreads=128,
        )
        r, xi = result(return_sep=True)

        print(f"Time elapsed for node{node_idx}: {time.time() - t0i:.2f} s")
        if ng_fixed:
            outfile = f"{OUTDATAPATH}/TPCF_train_node{node_idx}_{n_bins}bins_ng_fixed.npy"
        else:
            outfile = f"{OUTDATAPATH}/TPCF_train_node{node_idx}_{n_bins}bins.npy"
        # if Path(outfile).exists():
            # print(f"File {outfile} already exists, skipping...")
            # continue
        np.save(outfile, np.array([r, xi]))


    print(f"Total time elapsed: {time.time() - t0:.2f} s")

# def compute_TPCF_halomodel()

# compute_TPCF_fiducial_halocat(n_bins=64, threads=128)
# compute_TPCF_fiducial_halocat_halotools(n_bins=64, threads=128)
# compute_TPCF_train_halocats_halotools(n_bins=64)
compute_TPCF_train_halocats_pycorr(n_bins=64, ng_fixed=True)
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

def read_hdf5_files():
    ng_list = []
    nc_list = []
    ns_list = []
    N_gal = []

    for flag in dataset_names:
        print(f"flag={flag}") 
        fff = h5py.File(f"{INDATAPATH}/halocat_{flag}.hdf5", "r")
        N_nodes = len(fff.keys())
        ng_flag = []
        nc_flag = []
        ns_flag = []

        for node_idx in range(N_nodes):
            # print(f"Running node{node_idx}...", end=" ")

            HOD_group = fff[f"node{node_idx}"] 

            ng_flag.append(HOD_group.attrs['ng'])
            nc_flag.append(HOD_group.attrs['nc'])
            ns_flag.append(HOD_group.attrs['ns'])
            N_gal.append(len(HOD_group['x'][:]))
            
        ng_list.append(ng_flag)
        nc_list.append(nc_flag)
        ns_list.append(ns_flag)
        # print(ng_flag)
        fff.close()

        print("   ng      |   nc      |   ns   ")
        for i in range(len(ng_flag)):
            # print(f"{ng_flag[i]:.4e} | {nc_flag[i]:.4e} | {ns_flag[i]:.4e}")
            print(f"N_gal, node{i}={N_gal[i]}")
        exit()

    # print("ng_list:", ng_list)
    # print("nc_list:", nc_list)
    # print("ns_list:", ns_list)
    print(f"ng mean +/- std: ")
    print(f" train: {np.mean(ng_list[0]):.4e} +/- {np.std(ng_list[0]):.4e}")
    print(f" test:  {np.mean(ng_list[1]):.4e} +/- {np.std(ng_list[1]):.4e}")
    print(f" val:   {np.mean(ng_list[2]):.4e} +/- {np.std(ng_list[2]):.4e}")



def read_csv_original():
    HOD = pd.read_csv(f"../example/v130_HOD.csv")
    N_central = len(HOD[HOD["galaxy_type"] == "central"])
    N_satellite = len(HOD[HOD["galaxy_type"] == "satellite"])
    N_total = len(HOD)
    # print(HOD.value_counts("galaxy_type"))
    print(f"Number of centrals: {N_central}")
    print(f"Number of satellites: {N_satellite}")
    print(f"Total number of galaxies: {N_total}")
    
    ng = N_total / 2000.0**3
    print(f"galaxy number density: {ng:.4e}")


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



def compute_TPCF_train_halocats_pycorr(n_bins=128):
    """
    Halo_file: halocatalogue file for training parameters 
     - halo_file.keys(): individual files, ['node0', 'node1', ..., 'nodeN']
     - halo_file.attrs.keys(): cosmological parameters, e.g. ['H0', 'Om0', 'lnAs', 'n_s', ...]
     - halo_file['nodex'].attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
     - halo_file['nodex'].keys(): catalogue data, e.g. ['host_radius', 'x', 'y', 'z', 'v_x', ...]
    """

    halo_file = h5py.File(f"{INDATAPATH}/halocat_train.hdf5", "r")
    N_nodes = len(halo_file.keys())
    print(f"Computing TPCF for {N_nodes} nodes...")
    t0 = time.time()
    for node_idx in range(N_nodes):
        node_catalogue = halo_file[f"node{node_idx}"]
        x = np.array(node_catalogue['x'][:])
        y = np.array(node_catalogue['y'][:])
        z = np.array(node_catalogue['z'][:])

        t0i = time.time()

        r_bin_edges = np.logspace(np.log10(0.1), np.log10(130), n_bins)
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

        outfile = f"{OUTDATAPATH}/TPCF_train_node{node_idx}_{n_bins}bins.npy"
        np.save(outfile, np.array([r, xi]))


    print(f"Total time elapsed: {time.time() - t0:.2f} s")



def compute_TPCF_train_halocats_halotools(n_bins=128):
    """
    Halo_file: halocatalogue file for training parameters 
     - halo_file.keys(): individual files, ['node0', 'node1', ..., 'nodeN']
     - halo_file.attrs.keys(): cosmological parameters, e.g. ['H0', 'Om0', 'lnAs', 'n_s', ...]
     - halo_file['nodex'].attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
     - halo_file['nodex'].keys(): catalogue data, e.g. ['host_radius', 'x', 'y', 'z', 'v_x', ...]
    """

    halo_file = h5py.File(f"{INDATAPATH}/halocat_train.hdf5", "r")
    N_nodes = len(halo_file.keys())
    print(f"Computing TPCF for {N_nodes} nodes...")
    t0 = time.time()
    for node_idx in range(N_nodes):
        node_catalogue = halo_file[f"node{node_idx}"]
        x = np.array(node_catalogue['x'][:])
        y = np.array(node_catalogue['y'][:])
        z = np.array(node_catalogue['z'][:])

        t0i = time.time()

        r_bin_edges = np.logspace(np.log10(0.1), np.log10(130), n_bins)
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

        outfile = f"{OUTDATAPATH}/TPCF_train_node{node_idx}_{n_bins}bins_halotools.npy"
        np.save(outfile, np.array([r, xi]))


    print(f"Total time elapsed: {time.time() - t0:.2f} s")


# read_hdf5_files()
# read_csv_original()
# compute_TPCF_fiducial_halocat(n_bins=64, threads=128)
# compute_TPCF_fiducial_halocat_halotools(n_bins=64, threads=128)
# compute_TPCF_train_halocats_halotools(n_bins=64)

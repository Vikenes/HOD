import numpy as np 
import h5py 
import sys 
import time 
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from pycorr import TwoPointCorrelationFunction




HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
FIDUCIAL_HALOCAT_PATH = f"{HOD_DATA_PATH}/HOD_data/"



r_bins_log = np.logspace(np.log10(0.01), np.log10(5), 40, endpoint=False)
r_bins_lin = np.linspace(5.0, 150.0, 75)
r_bins_TPCF = np.concatenate((r_bins_log, r_bins_lin))


def compute_TPCF_fiducial_halocat():
    """
    fff.keys(): catalogue data, e.g. ['host_mass', 'host_id', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z', ...]
    fff.attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
    
    """
    fff = h5py.File(f"{FIDUCIAL_HALOCAT_PATH}/halocat_fiducial.hdf5", "r")

    outfile = f"{HOD_DATA_PATH}/individual_HOD_parameter_variation/data_results/TPCF_fiducial.npy"

    x = np.array(fff['x'][:])
    y = np.array(fff['y'][:])
    z = np.array(fff['z'][:])

    # r_bin_edges = np.logspace(np.log10(0.1), np.log10(130), n_bins)
    t0 = time.time()
    result = TwoPointCorrelationFunction(
        mode='s',
        edges=r_bins_TPCF,
        data_positions1=np.array([x, y, z]),
        boxsize=2000.0,
        engine='corrfunc',
        nthreads=128,
    )

    r, xi = result(
        return_sep=True
    )
    duration = time.time() - t0


    np.save(outfile, np.array([r, xi]))

    return r, xi, duration


def compute_TPCF_individual_param_varied(subfolder, 
                                         vary_param='sigma_logM',
                                         overwrite=False):
    """
    Halo_file: halocatalogue file for training parameters 
     - halo_file.keys(): individual files, ['node0', 'node1', ..., 'nodeN']
     - halo_file.attrs.keys(): cosmological parameters, e.g. ['H0', 'Om0', 'lnAs', 'n_s', ...]
     - halo_file['nodex'].attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
     - halo_file['nodex'].keys(): catalogue data, e.g. ['host_radius', 'x', 'y', 'z', 'v_x', ...]
    """
    allowed_params = ['sigma_logM', 'log10M1', 'kappa', 'alpha']
    if vary_param not in allowed_params:
        raise ValueError(f"vary_param must be one of {allowed_params}")

    INDATAPATH = f"{HOD_DATA_PATH}/individual_HOD_parameter_variation/{subfolder}"

    OUTDATAPATH = Path(f"{INDATAPATH}/tpcf_data")
    OUTDATAPATH.mkdir(parents=True, exist_ok=True)

    halo_file = h5py.File(f"{INDATAPATH}/halocat_vary_{vary_param}.hdf5", "r")
    N_nodes = len(halo_file.keys())
   
    
    print(f"Computing TPCF for {vary_param}...")
    t0 = time.time()
    
    for node_idx in range(N_nodes):
        outfile = f"{OUTDATAPATH}/TPCF_vary_{vary_param}_node{node_idx}.npy"
        
        if Path(outfile).exists() and not overwrite:
            print(f"File {outfile} already exists, skipping...")
            continue

        node_catalogue = halo_file[f"node{node_idx}"]
        x = np.array(node_catalogue['x'][:])
        y = np.array(node_catalogue['y'][:])
        z = np.array(node_catalogue['z'][:])

        t0i = time.time()

        result = TwoPointCorrelationFunction(
            mode='s',
            edges=r_bins_TPCF,
            data_positions1=np.array([x, y, z]),
            boxsize=2000.0,
            engine='corrfunc',
            nthreads=128,
        )
        r, xi = result(return_sep=True)

        print(f"Time elapsed for node{node_idx}: {time.time() - t0i:.2f} s")
        np.save(outfile, np.array([r, xi]))


    print(f"Total time elapsed: {time.time() - t0:.2f} s")



def compute_TPCF_all(subfolder, overwrite=False):
    compute_TPCF_individual_param_varied(subfolder=subfolder, vary_param='sigma_logM', overwrite=overwrite)
    compute_TPCF_individual_param_varied(subfolder=subfolder, vary_param='log10M1', overwrite=overwrite)
    compute_TPCF_individual_param_varied(subfolder=subfolder, vary_param='kappa', overwrite=overwrite)
    compute_TPCF_individual_param_varied(subfolder=subfolder, vary_param='alpha', overwrite=overwrite)



if len(sys.argv) == 2 or len(sys.argv) == 3:
    subfolder = sys.argv[1]
    overwrite = bool(sys.argv[2]) if len(sys.argv) == 3 else False
    compute_TPCF_all(subfolder, overwrite=overwrite)
else:
    raise ValueError(f"Incorrect number of arguments: {len(sys.argv)}")


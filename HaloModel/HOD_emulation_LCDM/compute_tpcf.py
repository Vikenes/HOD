import numpy as np 
import h5py 
import time 
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from pycorr import TwoPointCorrelationFunction

DATA_PATH           = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
HOD_DATA_PATH       = f"{DATA_PATH}/TPCF_emulation"
HOD_CATALOGUES_PATH = f"{HOD_DATA_PATH}/HOD_catalogues"
OUTPUT_PATH         = f"{HOD_DATA_PATH}/corrfunc_arrays"

OLD_INDATAPATH      = f"{DATA_PATH}/HOD_data"


def compute_TPCF_fiducial_halocat(n_bins=128, threads=12):
    """
    fff.keys(): catalogue data, e.g. ['host_mass', 'host_id', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z', ...]
    fff.attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
    
    """
    fff = h5py.File(f"{OLD_INDATAPATH}/halocat_fiducial.hdf5", "r")


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





def make_emulation_TPCF_arrays_from_halocats(ng_fixed=False, emulate_copy=False):
    ### Change the function to only compute TPCF, i.e. return r, xi 
    ### Move actual storing to "make_tpcf_emulation_data_files.py"
    ### 
    ### 
    ### 



    """
    Halo_file: halocatalogue file for training parameters 
     - halo_file.keys(): individual files, ['node0', 'node1', ..., 'nodeN']
     - halo_file.attrs.keys(): cosmological parameters, e.g. ['H0', 'Om0', 'lnAs', 'n_s', ...]
     - halo_file['nodex'].attrs.keys(): HOD parameters, e.g. ['alpha', 'log10Mmin', 'ng', 'nc', ...]
     - halo_file['nodex'].keys(): catalogue data, e.g. ['host_radius', 'x', 'y', 'z', 'v_x', ...]
    """
    
    """
    Compute galaxy-galaxy TPCF from halocats
    Make set for train, test, val catalogues used for emulation
    if file exists, skip
    """
    
    dataset_names = ['train', 'test', 'val']

    for flag in dataset_names:

        # Get filename of halo catalogue 
        halocat_fname_suffix = f"{flag}"
        if ng_fixed:
            halocat_fname_suffix += "_ng_fixed"
        halocat_fname = f"halocat_{halocat_fname_suffix}"

        # Load halo catalogue
        halo_file   = h5py.File(f"{HOD_CATALOGUES_PATH}/{halocat_fname}.hdf5", "r")
        N_nodes     = len(halo_file.keys()) # Number of parameter samples used to make catalogue 

        # Set up separation bins. Using same bins as Cuesta-Lazaro et al. 
        r_bins_log  = np.logspace(np.log10(0.01), np.log10(5), 40, endpoint=False)
        r_bins_lin  = np.linspace(5.0, 150.0, 75)
        r_bin_edges = np.concatenate((r_bins_log, r_bins_lin))
        n_bins      = len(r_bin_edges)
        
        print(f"Computing TPCF for {N_nodes} nodes...")
        t0 = time.time()
        
        # Compute TPCF for each node
        for node_idx in range(N_nodes):
            if ng_fixed:
                outfile = Path(f"{OUTPUT_PATH}/TPCF_{flag}_node{node_idx}_ng_fixed.npy")
            else:
                outfile = Path(f"{OUTPUT_PATH}/TPCF_{flag}_node{node_idx}.npy")
            
            if outfile.exists():# and not emulate_copy:
                print(f"File {outfile} already exists, skipping...")
                continue
            
            # elif outfile.exists() and emulate_copy:
            #     outfile_data = np.load(outfile)
            #     emulate_outfile = outfile.parent.parent / "emulation_data" / outfile.name
            #     if not emulate_outfile.exists():
            #         print(f"Saving emulation copy of {outfile.parent.name}/{outfile.name} to ", end='') 
            #         print(f"{emulate_outfile.parent.name}/{emulate_outfile.name}...")
            #         np.save(emulate_outfile, outfile_data)
            #     else:
            #         print(f"Emulation copy of {outfile.name} already exists, skipping...")

            #     continue 

            # Load galaxy positions for node
            node_catalogue = halo_file[f"node{node_idx}"]
            x = np.array(node_catalogue['x'][:])
            y = np.array(node_catalogue['y'][:])
            z = np.array(node_catalogue['z'][:])

            t0i = time.time()

            # Compute TPCF
            result = TwoPointCorrelationFunction(
                mode            = 's',
                edges           = r_bin_edges,
                data_positions1 = np.array([x, y, z]),
                boxsize         = 2000.0,
                engine          = 'corrfunc',
                nthreads        = 128,
                )
            r, xi = result(return_sep=True)

            print(f"Time elapsed for node{node_idx}: {time.time() - t0i:.2f} s")

            # Save TPCF to file
            np.save(outfile, np.array([r, xi]))
            # if emulate_copy:
            #     emulate_outfile = outfile.parent.parent / "emulation_data" / outfile.name
            #     if not emulate_outfile.exists():
            #         np.save(emulate_outfile, np.array([r, xi]))



        print(f"Total time elapsed: {time.time() - t0:.2f} s")


# compute_TPCF_fiducial_halocat(n_bins=64, threads=128)
# compute_TPCF_fiducial_halocat_halotools(n_bins=64, threads=128)
# compute_TPCF_train_halocats_halotools(n_bins=64)
# compute_TPCF_halocats_pycorr(n_bins=115, ng_fixed=True, flag="test")
# compute_TPCF_halocats_pycorr(n_bins=115, ng_fixed=True, flag="test", emulate_copy=True)
make_emulation_TPCF_arrays_from_halocats(ng_fixed=True, emulate_copy=True)
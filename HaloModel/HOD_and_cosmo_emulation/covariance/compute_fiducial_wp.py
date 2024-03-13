import numpy as np 
import pandas as pd
import time
from pathlib import Path
import h5py 
from Corrfunc.theory.wp import wp


OUTPATH     = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/covariance_data_fiducial")
HOD_FILE    = Path(OUTPATH / "halocat_fiducial_ng_fixed.hdf5")
if not HOD_FILE.exists():
    raise FileNotFoundError(f"{HOD_FILE} does not exist. Run make_fiducial_HOD_catalogue_hdf5_files.py")


def compute_wp_of_s_z_from_HOD_catalogue(
        r_binedge:  np.ndarray,
        threads:    int         = 128,
        pi_max:     float       = 200.0,
        ):

    # Create outfile for wp 
    outfile = Path(OUTPATH / "wp_from_sz_fiducial_ng_fixed.hdf5")
    if outfile.exists():
        input(f"{outfile} exists. Press enter to overwrite.")
    wp_file         = h5py.File(outfile, "w")
    
    # Read HOD catalogue
    HOD_catalogue   = h5py.File(HOD_FILE, "r")

    # Lists to compute wp mean and std
    wp_lst = []
    rp_lst = []

    for key in HOD_catalogue.keys():
        if not key.startswith("Abacus"):
            # Only use the AbacusSummit simulations
            continue

        t0          = time.time()
        HOD_group   = HOD_catalogue[key]

        # Compute wp from galaxy positions from halo catalogue
        result_wp = wp(
            boxsize     = 2000.0,
            pimax       = pi_max,
            nthreads    = threads,
            binfile     = r_binedge,
            X           = np.array(HOD_group['x'][:]),
            Y           = np.array(HOD_group['y'][:]),
            Z           = np.array(HOD_group['s_z'][:]),
            output_rpavg=True,
        )
        
        # Get perpendicular separation bins and wp 
        r_perp = result_wp["rpavg"]
        w_p    = result_wp["wp"]
       
        # Save wp to file
        wp_group = wp_file.create_group(key)
        wp_group.create_dataset("r_perp", data=r_perp)
        wp_group.create_dataset("w_p",    data=w_p)

        # Store wp for mean and std, and rp for mean
        wp_lst.append(w_p)
        rp_lst.append(r_perp)

        print(f"Done with {key}. Took {time.time() - t0:.2f} s")
    
    # Compute mean and std of wp 
    wp_all      = np.array(wp_lst)
    wp_mean     = np.mean(wp_all, axis=0)
    wp_stddev   = np.std(wp_all, axis=0)
    wp_file.create_dataset("wp_mean",    data=wp_mean)
    wp_file.create_dataset("wp_stddev", data=wp_stddev)

    # Compute mean of rperp
    rp_mean = np.mean(np.array(rp_lst), axis=0)
    wp_file.create_dataset("rp_mean", data=rp_mean)

    wp_file.close()
    HOD_catalogue.close()


compute_wp_of_s_z_from_HOD_catalogue(r_binedge = np.geomspace(0.5, 60, 30))
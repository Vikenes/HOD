import numpy as np 
import pandas as pd
import time
from pathlib import Path
import h5py 
from Corrfunc.theory.wp import wp
# from Corrfunc.theory.xi import xi as TwoPointCorrelationFunction
from pycorr import TwoPointCorrelationFunction


OUTPATH     = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data/MGGLAM")
HOD_FILE    = Path(OUTPATH / "halocat_fiducial_ng_fixed.hdf5")
if not HOD_FILE.exists():
    raise FileNotFoundError(f"{HOD_FILE} does not exist. Run make_HOD.py first")


def compute_wp_of_s_z_from_HOD_catalogue(
        r_binedge:  np.ndarray,
        threads:    int         = 128,
        pi_max:     float       = 200.0,
        ):

    # Create outfile for wp 
    outfile = Path(OUTPATH / "wp_from_sz_fiducial_ng_fixed.hdf5")
    if outfile.exists():
        input(f"{outfile} exists. Press enter to overwrite.")
    print(f"Computing wp for {outfile}")
    wp_file         = h5py.File(outfile, "w")
    
    # Read HOD catalogue
    HOD_catalogue   = h5py.File(HOD_FILE, "r")

    # Lists to compute wp mean and std
    wp_lst = []
    rp_lst = []

    for key in HOD_catalogue.keys():

        t0          = time.time()
        HOD_group   = HOD_catalogue[key]

        # Compute wp from galaxy positions from halo catalogue
        result_wp = wp(
            boxsize     = 1000.0,
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


def compute_tpcf_from_HOD_catalogue(
        r_binedge:  np.ndarray,
        r_min:      float       = 0.6,
        r_max:      float       = 100.0,
        use_r_mask: bool        = True,
        threads:    int         = 128,
        ):
    
    # Create outfile for tpcf
    outfile = Path(OUTPATH / "tpcf_r_fiducial_ng_fixed.hdf5")
    if outfile.exists():
        input(f"{outfile} exists. Press enter to overwrite.")
    print(f"Computing TPCF for {outfile}")

    tpcf_file       = h5py.File(outfile, "w")
    
    # Read HOD catalogue
    HOD_catalogue   = h5py.File(HOD_FILE, "r")

    # Lists to compute wp mean and std
    tpcf_lst = []
    r_lst = []

    for key in HOD_catalogue.keys():

        t0          = time.time()
        HOD_group   = HOD_catalogue[key]

        X           = np.array(HOD_group['x'][:])
        Y           = np.array(HOD_group['y'][:])
        Z           = np.array(HOD_group['z'][:])

        # Compute wp from galaxy positions from halo catalogue
        result_tpcf = TwoPointCorrelationFunction(
            mode            = 's',
            edges           = r_binedge,
            data_positions1 = np.array([X, Y, Z]),
            boxsize         = 1000.0,
            engine          = "corrfunc",
            nthreads        = threads,
        )

        r, xi = result_tpcf(return_sep=True)
        if use_r_mask:
            r_mask  = (r > r_min) & (r < r_max)
            r       = r[r_mask]
            xi      = xi[r_mask]

        
        # Save wp to file
        tpcf_group = tpcf_file.create_group(key)
        tpcf_group.create_dataset("r", data=r)
        tpcf_group.create_dataset("xi",data=xi)

        # Store wp for mean and std, and rp for mean
        r_lst.append(r)
        tpcf_lst.append(xi)

        print(f"Done with {key}. Took {time.time() - t0:.2f} s")
    
    # Compute mean and std of wp 
    tpcf_all      = np.array(tpcf_lst)
    tpcf_mean     = np.mean(tpcf_all, axis=0)
    tpcf_stddev   = np.std(tpcf_all, axis=0)
    tpcf_file.create_dataset("xi_mean",   data=tpcf_mean)
    tpcf_file.create_dataset("xi_stddev", data=tpcf_stddev)

    # Compute mean of rperp
    r_mean = np.mean(np.array(r_lst), axis=0)
    tpcf_file.create_dataset("r_mean", data=r_mean)

    tpcf_file.close()
    HOD_catalogue.close()


compute_wp_of_s_z_from_HOD_catalogue(r_binedge = np.geomspace(0.5, 40, 40))

r_bin_edges = np.concatenate((
    np.logspace(np.log10(0.01), np.log10(5), 40, endpoint=False),
    np.linspace(5.0, 150.0, 75)
    ))

compute_tpcf_from_HOD_catalogue(r_binedge = r_bin_edges, use_r_mask=True)
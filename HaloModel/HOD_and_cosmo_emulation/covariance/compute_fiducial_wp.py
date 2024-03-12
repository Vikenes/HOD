import numpy as np 
import pandas as pd
import time
from pathlib import Path
import h5py 
from Corrfunc.theory.wp import wp


D13_BASE_PATH       = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
D13_EMULATION_PATH  = Path(f"{D13_BASE_PATH}/emulation_files/fiducial_data")

OUTPATH  = Path(D13_EMULATION_PATH / "wp_data")
HOD_FILE = Path(D13_EMULATION_PATH / "HOD_catalogues/halocat_fiducial_ng_fixed.hdf5")

r_binedge = np.geomspace(0.5, 60, 30)


def compute_wp_of_s_z_from_HOD_catalogue(
        threads:    int   = 128,
        pi_max:     float = 200.0,
        ):

    outfile = Path(OUTPATH / "wp_from_sz_fiducial_ng_fixed.hdf5")
    if not outfile.exists():
        print(f"{outfile} exists. Skipping.")
        return
    
    HOD_catalogue = h5py.File(HOD_FILE, "r")
    fff = h5py.File(outfile, "w")

    wp_lst = []
    rp_lst = []


    for key in HOD_catalogue.keys():
        HOD_group = HOD_catalogue[key]
        t0 = time.time()
        # Load galaxy positions for node from halo catalogue
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

        r_perp = result_wp["rpavg"]
        w_p    = result_wp["wp"]
       
        # Save TPCF to file
        wp_group = fff.create_group(key)
        wp_group.create_dataset("r_perp", data=r_perp)
        wp_group.create_dataset("w_p",    data=w_p)

        wp_lst.append(w_p)
        rp_lst.append(r_perp)

        print(f"Done with {key}. Took {time.time() - t0:.2f} s")
    
    wp_all = np.array(wp_lst)
    wp_mean = np.mean(
        wp_all,
        axis=0,
    )
    wp_stddev = np.std(
        wp_all,
        axis=0,
    )
    fff.create_dataset("wp_mean",    data=wp_mean)
    fff.create_dataset("wp_stddev", data=wp_stddev)

    rp_mean = np.mean(
        np.array(rp_lst),
        axis=0,
    )
    fff.create_dataset("rp_mean", data=rp_mean)

    fff.close()
    HOD_catalogue.close()

compute_wp_of_s_z_from_HOD_catalogue()
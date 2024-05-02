import numpy as np 
import h5py 
import time 
from pathlib import Path
from Corrfunc.theory.wp import wp as compute_wp


D13_DATA_PATH           = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files")
assert D13_DATA_PATH.exists(), f"Error: {D13_DATA_PATH} not found. Check path."

"""
FIX FLAGS BEING LOOPED OVER  
"""

class WP_ABACUS:
    def __init__(
            self,
            rperp_binedges:     np.array,
            ng_fixed:           bool,
            boxsize:            float = 2000.0,
            pi_max:             float = 200.0,
            nthreads:           int   = 128,
            ):
        
        self.rperp_binedges = rperp_binedges
        self.ng_fixed       = ng_fixed
        self.boxsize        = boxsize
        self.pi_max         = pi_max
        self.nthreads       = nthreads
        

        if ng_fixed:
            self.ng_suffix = "_ng_fixed"
        else:
            self.ng_suffix = ""

    
    
    def save_wp_data_sz_from_HOD_catalogue(
            self,
            version: int,
            phase:   int,
            flags:   list,
    ):
        SIM_DATA_PATH       = Path(D13_DATA_PATH / f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}")
        HOD_CATALOGUE_PATH  = Path(SIM_DATA_PATH / "HOD_catalogues")
        OUTPUT_PATH         = Path(SIM_DATA_PATH / "wp_data")
        
        Path(OUTPUT_PATH).mkdir(parents=False, exist_ok=True)
        fname_suffix    = [f"{flag}{self.ng_suffix}" for flag in flags]
        halocat_fnames  = [f"halocat_{suffix}.hdf5" for suffix in fname_suffix]
        wp_fnames     = [f"wp_{suffix}.hdf5" for suffix in fname_suffix]

        for wp_fname, halocat_fname in zip(wp_fnames, halocat_fnames):

            OUTFILE     = f"{OUTPUT_PATH}/{wp_fname}"
            if Path(OUTFILE).exists():
                print(f"File {OUTFILE} already exists, skipping...")
                continue

            halo_filename = Path(f"{HOD_CATALOGUE_PATH}/{halocat_fname}")
            if not halo_filename.exists():
                # No HOD catalogue with "train" data for c000_ph000-ph024 or c001-c004
                continue 

            halo_file   = h5py.File(halo_filename, "r")
            N_nodes     = len(halo_file.keys()) # Number of parameter samples used to make catalogue

            print(f"Computing all {N_nodes} {wp_fname} for {SIM_DATA_PATH.name}...")

            t0 = time.time()
            fff = h5py.File(OUTFILE, "w")
            # Compute wp for each node
            for node_idx in range(N_nodes):
                # Load galaxy positions for node from halo catalogue
                HOD_node_catalogue = halo_file[f"node{node_idx}"]
                x = np.array(HOD_node_catalogue['x'][:])
                y = np.array(HOD_node_catalogue['y'][:])
                z = np.array(HOD_node_catalogue['s_z'][:])
                
                result_wp = compute_wp(
                    boxsize     = 2000.0,
                    pimax       = self.pi_max,
                    nthreads    = self.nthreads,
                    binfile     = self.rperp_binedges,
                    X           = np.array(HOD_node_catalogue['x'][:]),
                    Y           = np.array(HOD_node_catalogue['y'][:]),
                    Z           = np.array(HOD_node_catalogue['s_z'][:]),
                    output_rpavg=True,
                )
                r_perp = result_wp["rpavg"]
                wp     = result_wp["wp"]

                # Save wp to file
                node_group = fff.create_group(f'node{node_idx}')
                node_group.create_dataset("r_perp",  data=r_perp)
                node_group.create_dataset("wp", data=wp)

            print(f"Done. Took {time.time() - t0:.2f} s")
            fff.close()
            halo_file.close()
        

    def save_wp_c000_phases(self, start=0, stop=25):
        assert start >= 0 and stop <= 25 and start < stop, "Invalid phase range for c000"
        phases = np.arange(start, stop)
        for phase in phases:
            self.save_wp_data_sz_from_HOD_catalogue(
                version = 0, 
                phase   = phase,
                flags   = ["test"])
            
    def save_wp_c001_c004(self, start=1, stop=5):
        assert start >= 1 and stop <= 5 and start < stop, "Invalid version range for c001-c004" 
        versions = np.arange(start, stop)
        for version in versions:
            self.save_wp_data_sz_from_HOD_catalogue(
                version = version, 
                phase   = 0,
                flags   = ["test"])



# Use same bins as Cuesta-Lazaro et al.
rperp_binedges = np.geomspace(0.5, 40, 40)

# Instantiate class
WP = WP_ABACUS(rperp_binedges, ng_fixed=False)


# WP.save_wp_c000_phases(start=0, stop=5)
# WP.save_wp_c000_phases(start=5, stop=10)
# WP.save_wp_c000_phases(start=10, stop=15)
# WP.save_wp_c000_phases(start=15, stop=20)
# WP.save_wp_c000_phases(start=20, stop=25)
# WP.save_wp_c001_c004(start=1, stop=5)

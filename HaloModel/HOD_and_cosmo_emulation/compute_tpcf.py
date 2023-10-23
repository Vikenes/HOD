import numpy as np 
import h5py 
import time 
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from pycorr import TwoPointCorrelationFunction

D13_BASE_PATH           = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"

# HOD_DATA_PATH       = f"{DATA_PATH}/TPCF_emulation"
# HOD_CATALOGUES_PATH = f"{HOD_DATA_PATH}/HOD_catalogues"
# OUTPUT_PATH         = f"{HOD_DATA_PATH}/corrfunc_arrays"

class TPCF_ABACUS:
    def __init__(
            self,
            r_bin_edges: np.array,
            ng_fixed:    bool = True,
            boxsize:     float = 2000.0,
            engine:      str   = 'corrfunc',
            nthreads:    int   = 128,
            ):
        self.r_bin_edges = r_bin_edges
        self.ng_fixed    = ng_fixed
        self.boxsize     = boxsize
        self.engine      = engine
        self.nthreads    = nthreads
        

        dataset_names = ['train', 'test', 'val']
        if ng_fixed:
            ng_suffix = "_ng_fixed"
        else:
            ng_suffix = ""
        self.fname_suffix    = [f"{flag}{ng_suffix}" for flag in dataset_names]
        self.halocat_fnames  = [f"halocat_{suffix}.hdf5" for suffix in self.fname_suffix]
        self.TPCF_fnames     = [f"TPCF_{suffix}.hdf5" for suffix in self.fname_suffix]

        # TO BE DECIDED:
        # Save one TPCF array per node per catalogue
        # or one large TPCF hdf5 file for each catalogue?
        

    def compute_TPCF_from_gal_pos(
            self,
            galaxy_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute TPCF for a set of galaxy positions
        with shape (3, N) where N is the number of galaxies
        i.e. galaxy_positions = np.array([x, y, z])
        """
        
        assert galaxy_positions.shape[0] == 3, "galaxy_positions must be a 3xN array"
        assert galaxy_positions.ndim == 2, "galaxy_positions must be a 3xN array"


        
        result = TwoPointCorrelationFunction(
                mode            = 's',
                edges           = self.r_bin_edges,
                data_positions1 = galaxy_positions,
                boxsize         = self.boxsize,
                engine          = self.engine,
                nthreads        = self.nthreads,
                )
        r, xi = result(return_sep=True)
        return np.array([r, xi])
    
    def save_TPCF_data_from_HOD_catalogue(
            self,
            version:    int  = 0,
            phase:      int  = 0,
    ):
        SIMNAME             = f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}"
        SIM_DATA_PATH       = f"{D13_BASE_PATH}/emulation_files/{SIMNAME}"
        HOD_CATALOGUE_PATH  = f"{SIM_DATA_PATH}/HOD_catalogues"
        OUTPUT_PATH         = f"{SIM_DATA_PATH}/TPCF_data"
        
        if not Path(HOD_CATALOGUE_PATH).exists():
            print(f"Error: HOD catalogue required to make TPCF files.")
            print(f" - Directory {HOD_CATALOGUE_PATH} not found, aborting...")
            raise FileNotFoundError
        else:
            # Create output directory if it doesn't exist
            Path(OUTPUT_PATH).mkdir(parents=False, exist_ok=True)

        for TPCF_fname, halocat_fname in zip(self.TPCF_fnames, self.halocat_fnames):

            OUTFILE     = f"{OUTPUT_PATH}/{TPCF_fname}"

            if Path(OUTFILE).exists():
                print(f"File {OUTFILE} already exists, skipping...")
                continue

            halo_file   = h5py.File(f"{HOD_CATALOGUE_PATH}/{halocat_fname}", "r")
            N_nodes     = len(halo_file.keys()) # Number of parameter samples used to make catalogue

            print(f"Computing all {N_nodes} {TPCF_fname} for {SIMNAME}...", end=" ")
            t0 = time.time()
            fff = h5py.File(OUTFILE, "w")
            # Compute TPCF for each node
            for node_idx in range(N_nodes):
                # Load galaxy positions for node from halo catalogue
                HOD_node_catalogue = halo_file[f"node{node_idx}"]
                x = np.array(HOD_node_catalogue['x'][:])
                y = np.array(HOD_node_catalogue['y'][:])
                z = np.array(HOD_node_catalogue['z'][:])
                
                # Compute TPCF 
                r, xi = self.compute_TPCF_from_gal_pos(
                    galaxy_positions=np.array([x, y, z])
                    )

                # Save TPCF to file
                node_group = fff.create_group(f'node{node_idx}')
                node_group.create_dataset("r",  data=r)
                node_group.create_dataset("xi", data=xi)

            print(f"Done. Took {time.time() - t0:.2f} s")
            fff.close()
            halo_file.close()
        
    def save_TPCF_all_versions(
            self,
            c000_phases:            bool = False,
            c001_c004:              bool = False,
            linear_derivative_grid: bool = False,
            broad_emulator_grid:    bool = False,
        ):
        """
        With nthreads=128, each TPCF takes roughly 1s to compute. 
        However, with several hundreds needing to be computed for each version,
        there is a lot of overhead, and it seems to not activate each node fully before completion. 
        It might therefore be faster to compute the TPCF's in parallel for N versions 
        simultaneously, using nthreads=128/N to compute the TPCF's. 
        However, one should be cautious, to ensure that file writing is done correctly.  
        """
        if c000_phases:
            phases   = np.arange(0, 25)
            for phase in phases:
                self.save_TPCF_data_from_HOD_catalogue(
                    version = 0, 
                    phase   = phase)
            
        if c001_c004:
            versions = np.arange(1, 5)
            for version in versions:
                self.save_TPCF_data_from_HOD_catalogue(
                    version = version, 
                    phase   = 0)
        
        if linear_derivative_grid:
            versions = np.arange(100, 127)
            for version in versions:
                self.save_TPCF_data_from_HOD_catalogue(
                    version = version, 
                    phase   = 0)
        
        if broad_emulator_grid:
            versions = np.arange(130, 182)
            for version in versions:
                self.save_TPCF_data_from_HOD_catalogue(
                    version = version, 
                    phase   = 0)

# Use same bins as Cuesta-Lazaro et al.
r_bin_edges = np.concatenate((
    np.logspace(np.log10(0.01), np.log10(5), 40, endpoint=False),
    np.linspace(5.0, 150.0, 75)
    ))

            
tt = TPCF_ABACUS(
    r_bin_edges=r_bin_edges, 
    ng_fixed=True,
    nthreads=128
    )

#tt.save_TPCF_all_versions(
#    c000_phases=False,
#    c001_c004=False,
#    linear_derivative_grid=True,
#    broad_emulator_grid=False,
#)

tt.save_TPCF_all_versions(
    c000_phases=False,
    c001_c004=False,
    linear_derivative_grid=False,
    broad_emulator_grid=True,
)
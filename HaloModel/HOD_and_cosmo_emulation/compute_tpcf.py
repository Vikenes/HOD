import numpy as np 
import h5py 
import time 
from pathlib import Path
import concurrent.futures
from pycorr import TwoPointCorrelationFunction

D13_DATA_PATH           = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files")
assert D13_DATA_PATH.exists(), f"Error: {D13_DATA_PATH} not found. Check path."

"""
FIX FLAGS BEING LOOPED OVER  
"""

class TPCF_ABACUS:
    def __init__(
            self,
            r_bin_edges: np.array,
            ng_fixed:    bool,
            use_sep_avg: bool,
            boxsize:     float = 2000.0,
            engine:      str   = 'corrfunc',
            nthreads:    int   = 128,
            ):
        
        self.r_bin_edges    = r_bin_edges
        self.r_bin_centers  = (r_bin_edges[1:] + r_bin_edges[:-1]) / 2.0
        self.ng_fixed       = ng_fixed
        self.boxsize        = boxsize
        self.engine         = engine
        self.nthreads       = nthreads
        self.process_TPCF   = self.compute_TPCF_from_gal_pos_with_sepavg if use_sep_avg else self.compute_TPCF_from_gal_pos_without_sepavg 
        

        dataset_names = ['train', 'test', 'val']
        if ng_fixed:
            self.ng_suffix = "_ng_fixed"
        else:
            self.ng_suffix = ""

    def compute_TPCF_from_gal_pos_with_sepavg(
            self,
            galaxy_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute TPCF for a set of galaxy positions
        with shape (3, N) where N is the number of galaxies
        i.e. galaxy_positions = np.array([x, y, z])
        Return average separation distance in each bin 
        """
        
        result = TwoPointCorrelationFunction(
                mode            = 's',
                edges           = self.r_bin_edges,
                data_positions1 = galaxy_positions,
                boxsize         = self.boxsize,
                engine          = self.engine,
                nthreads        = self.nthreads,
                )

        return result(return_sep=True) 
    
    def compute_TPCF_from_gal_pos_without_sepavg(
            self,
            galaxy_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute TPCF for a set of galaxy positions
        with shape (3, N) where N is the number of galaxies
        i.e. galaxy_positions = np.array([x, y, z])
        """

        
        result = TwoPointCorrelationFunction(
                mode            = 's',
                edges           = self.r_bin_edges,
                data_positions1 = galaxy_positions,
                boxsize         = self.boxsize,
                engine          = self.engine,
                nthreads        = self.nthreads,
                )

        return self.r_bin_centers, result(return_sep=False) 
    
    
    def save_TPCF_data_from_HOD_catalogue(
            self,
            version: int,
            phase:   int,
            flags:   list,
    ):
        SIM_DATA_PATH       = Path(D13_DATA_PATH / f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}")
        HOD_CATALOGUE_PATH  = Path(SIM_DATA_PATH / "HOD_catalogues")
        OUTPUT_PATH         = Path(SIM_DATA_PATH / "TPCF_data")
        
        Path(OUTPUT_PATH).mkdir(parents=False, exist_ok=False)
        fname_suffix    = [f"{flag}{self.ng_suffix}" for flag in flags]
        halocat_fnames  = [f"halocat_{suffix}.hdf5" for suffix in fname_suffix]
        TPCF_fnames     = [f"TPCF_{suffix}.hdf5" for suffix in fname_suffix]

        for TPCF_fname, halocat_fname in zip(TPCF_fnames, halocat_fnames):

            OUTFILE     = f"{OUTPUT_PATH}/{TPCF_fname}"
            if Path(OUTFILE).exists():
                print(f"File {OUTFILE} already exists, skipping...")
                continue

            halo_filename = Path(f"{HOD_CATALOGUE_PATH}/{halocat_fname}")
            if not halo_filename.exists():
                # No HOD catalogue with "train" data for c000_ph000-ph024 or c001-c004
                continue 

            halo_file   = h5py.File(halo_filename, "r")
            N_nodes     = len(halo_file.keys()) # Number of parameter samples used to make catalogue

            print(f"Computing all {N_nodes} {TPCF_fname} for {SIM_DATA_PATH.name}...")

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
                r, xi = self.process_TPCF(
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
            c000_phases:            bool,
            c001_c004:              bool,
            linear_derivative_grid: bool,
            broad_emulator_grid:    bool,
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
                    phase   = phase,
                    flags   = ["test", "val"])
            
        if c001_c004:
            versions = np.arange(1, 5)
            for version in versions:
                self.save_TPCF_data_from_HOD_catalogue(
                    version = version, 
                    phase   = 0,
                    flags   = ["test", "val"])
        
        if linear_derivative_grid:
            versions = np.arange(100, 127)
            for version in versions:
                self.save_TPCF_data_from_HOD_catalogue(
                    version = version, 
                    phase   = 0,
                    flags   = ["train", "test", "val"])
        
        if broad_emulator_grid:
            versions = np.arange(130, 182)
            for version in versions:
                self.save_TPCF_data_from_HOD_catalogue(
                    version = version, 
                    phase   = 0,
                    flags   = ["train", "test", "val"])
                
    def save_TPCF_linear_derivative_grid(self, start=100, stop=127):
        versions = np.arange(start, stop)
        for version in versions:
            self.save_TPCF_data_from_HOD_catalogue(
                version = version, 
                phase   = 0,
                flags   = ["train", "test", "val"])

    def save_TPCF_broad_emulator_grid(self, start=130, stop=182):
        versions = np.arange(start, stop)
        for version in versions:
            self.save_TPCF_data_from_HOD_catalogue(
                version = version, 
                phase   = 0,
                flags   = ["train", "test", "val"])



def run_c000_parallel(r_bin_edges, nthreads):
    tt = TPCF_ABACUS(
        r_bin_edges=r_bin_edges, 
        ng_fixed=False,
        nthreads=nthreads,
        use_sep_avg=True,
        )
    phases   = np.arange(0, 25)
    t000 = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(
            tt.save_TPCF_data_from_HOD_catalogue, 
            [0 for phase in phases],
            phases,
            [["test", "val"] for phase in phases]
            )
    print(f"Done with c000. Took {time.time() - t000:.2f} s")
        
    

def run_c001_c004_parallel(r_bin_edges, nthreads=32):
    tt = TPCF_ABACUS(
        r_bin_edges=r_bin_edges, 
        ng_fixed=False,
        nthreads=nthreads,
        use_sep_avg=True,
        )
    versions   = np.arange(1, 5)
    t000 = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(
            tt.save_TPCF_data_from_HOD_catalogue, 
            versions,
            [0 for _ in versions],
            [["test", "val"] for _ in versions]
            )
        
    print(f"Done with c001-c004. Took {time.time() - t000:.2f} s")


def run_lin_der_grid_parallel(r_bin_edges, start, stop, nthreads):
    tt = TPCF_ABACUS(
        r_bin_edges=r_bin_edges, 
        ng_fixed=False,
        nthreads=nthreads,
        use_sep_avg=True,
        )
    assert start >= 100 and stop <= 127 and start < stop, f"Error: start={start} and stop={stop} not valid for linear derivative grid."
    versions   = np.arange(start, stop)
    t000 = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(
            tt.save_TPCF_data_from_HOD_catalogue, 
            versions,
            [0 for _ in versions],
            [["train", "test", "val"] for _ in versions]
            )
        
    print(f"Done with c{start}-c{stop}. Took {time.time() - t000:.2f} s")

# Use same bins as Cuesta-Lazaro et al.
r_bin_edges = np.concatenate((
    np.logspace(np.log10(0.01), np.log10(5), 40, endpoint=False),
    np.linspace(5.0, 150.0, 75)
    ))

            
tt = TPCF_ABACUS(
    r_bin_edges=r_bin_edges, 
    ng_fixed=False,
    nthreads=128,
    use_sep_avg=True,
    )


# tt.save_TPCF_linear_derivative_grid(108, 127)
# tt.save_TPCF_broad_emulator_grid(130, 156)
# tt.save_TPCF_broad_emulator_grid(156, 182)


# tt.save_TPCF_all_versions(
#     c000_phases            = False,
#     c001_c004              = False,
#     linear_derivative_grid = True,
#     broad_emulator_grid    = False,
#     )



# run_c000_parallel(r_bin_edges)
# run_c001_c004_parallel(r_bin_edges)
# run_lin_der_grid_parallel(r_bin_edges, 100, 104, nthreads=32)
# run_lin_der_grid_parallel(r_bin_edges, 104, 106, nthreads=64)
# run_lin_der_grid_parallel(r_bin_edges, 104, 108, nthreads=64)


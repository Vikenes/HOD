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
        
        Path(OUTPUT_PATH).mkdir(parents=False, exist_ok=True)
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
        

    def save_TPCF_c000_phases(self, start=0, stop=25):
        assert start >= 0 and stop <= 25 and start < stop, "Invalid phase range for c000"
        phases = np.arange(start, stop)
        for phase in phases:
            self.save_TPCF_data_from_HOD_catalogue(
                version = 0, 
                phase   = phase,
                flags   = ["test"])
            
    def save_TPCF_c001_c004(self, start=1, stop=5):
        assert start >= 1 and stop <= 5 and start < stop, "Invalid version range for c001-c004" 
        versions = np.arange(start, stop)
        for version in versions:
            self.save_TPCF_data_from_HOD_catalogue(
                version = version, 
                phase   = 0,
                flags   = ["test"])


    def save_TPCF_linear_derivative_grid(self, start=100, stop=127):
        assert start >= 100 and stop <= 127 and start < stop, "Invalid version range for linear derivative grid"
        versions = np.arange(start, stop)
        for version in versions:
            self.save_TPCF_data_from_HOD_catalogue(
                version = version, 
                phase   = 0,
                flags   = ["train", "val"])

    def save_TPCF_broad_emulator_grid(self, start=130, stop=182):
        assert start >= 130 and stop <= 182 and start < stop, "Invalid version range for broad emulator grid"
        versions = np.arange(start, stop)
        T0 = time.time()
        for version in versions:
            self.save_TPCF_data_from_HOD_catalogue(
                version = version, 
                phase   = 0,
                flags   = ["train", "val"])
        dur = time.time() - T0
        print(f"Total time for {start}-{stop}: {dur//60:.0f} min {dur%60:.1f} s")



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

# tt.save_TPCF_c000_phases(0, 3)
# tt.save_TPCF_c000_phases(3, 6)
# tt.save_TPCF_c000_phases(6, 9)
# tt.save_TPCF_c000_phases(9, 12)
# tt.save_TPCF_c000_phases(12, 15)
# tt.save_TPCF_c000_phases(15, 20)
# tt.save_TPCF_c000_phases(20, 25)

# tt.save_TPCF_c001_c004(1, 5)

# tt.save_TPCF_linear_derivative_grid(100, 105)
# tt.save_TPCF_linear_derivative_grid(105, 110)
# tt.save_TPCF_linear_derivative_grid(110, 115)
# tt.save_TPCF_linear_derivative_grid(115, 117)
# tt.save_TPCF_linear_derivative_grid(117, 119)
# tt.save_TPCF_linear_derivative_grid(119, 121)
# tt.save_TPCF_linear_derivative_grid(121, 123)
# tt.save_TPCF_linear_derivative_grid(123, 125)
# tt.save_TPCF_linear_derivative_grid(125, 127)


# tt.save_TPCF_broad_emulator_grid(130, 135)
# tt.save_TPCF_broad_emulator_grid(135, 140)
# tt.save_TPCF_broad_emulator_grid(140, 145)
# tt.save_TPCF_broad_emulator_grid(145, 147)
# tt.save_TPCF_broad_emulator_grid(147, 149)
# tt.save_TPCF_broad_emulator_grid(149, 151)
# tt.save_TPCF_broad_emulator_grid(151, 153)
# tt.save_TPCF_broad_emulator_grid(153, 155)
# tt.save_TPCF_broad_emulator_grid(155, 157)
# tt.save_TPCF_broad_emulator_grid(157, 159)
# tt.save_TPCF_broad_emulator_grid(159, 161)
# tt.save_TPCF_broad_emulator_grid(161, 163)
# tt.save_TPCF_broad_emulator_grid(163, 165)
# tt.save_TPCF_broad_emulator_grid(165, 167)
# tt.save_TPCF_broad_emulator_grid(167, 169)
# tt.save_TPCF_broad_emulator_grid(169, 171)
# tt.save_TPCF_broad_emulator_grid(171, 173)
# tt.save_TPCF_broad_emulator_grid(173, 175)
# tt.save_TPCF_broad_emulator_grid(175, 177)
# tt.save_TPCF_broad_emulator_grid(177, 179)
# tt.save_TPCF_broad_emulator_grid(179, 182)

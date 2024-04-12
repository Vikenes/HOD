import asdf 
import numpy as np 
import time 
from pathlib import Path

import concurrent.futures
from itertools import repeat

N_files_per_version = 34
BASEPATH        = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit")
ABACUSPATH      = Path(BASEPATH / "small") # Location of data 
EMULATION_PATH  = Path(BASEPATH / "emulation_files/small") # Location of data
OUTDIR          = Path("pos_vel_mass_arrays") # name of directory where output arrays are stored 


# Get default header info from c000_ph3000 
af_filename = Path(ABACUSPATH / "AbacusSummit_small_c000_ph3000/halos/z0.250/halo_info/halo_info_000.asdf")
af          = asdf.open(af_filename)
ParticleMassHMsun  = af['header']['ParticleMassHMsun']
BoxSizeHMpc        = af['header']['BoxSizeHMpc']
VelZSpace_to_kms   = af['header']['VelZSpace_to_kms']
af.close()


def save_L1_pos_vel_mass_single_version(
        phase:      int     = 3000,
        ):
    """
    Create arrays of L1 data for all files in each simulation version.
    Outputs: 
        pos(N,3): central coordinates 
        vel(N,3): CoM velocities
        mass(N) : Total mass
    Resulting array is saved to disk.  
    """
    simname = Path(f"AbacusSummit_small_c000_ph{phase}")
    simulation  = Path(ABACUSPATH / simname)
    if not simulation.is_dir():
        print(f"Simulation {simname} doesn' exist. Continuing...")
        return 

    af_filename = Path(f"{simulation}/halos/z0.250/halo_info/halo_info_000.asdf")
    
    # Create output directory if it does not exist
    L1_path = Path(EMULATION_PATH / simname / OUTDIR)
    L1_path.mkdir(parents=True, exist_ok=True)

    # Create output filenames
    pos_fname  = Path(L1_path / "L1_pos.npy")
    vel_fname  = Path(L1_path / "L1_vel.npy")
    mass_fname = Path(L1_path / "L1_mass.npy")

    # Check if arrays already exist, if so, skip
    if pos_fname.exists() and vel_fname.exists() and mass_fname.exists():
        print(f"Arrays for {simname} already exist, skipping")
        return
    
    # Start computation    
    print(f"Saving L1 qtys for simulation {simname}")
    t0 = time.time()

    af          = asdf.open(af_filename)
    ### CU = code units
    L1_pos   = af['data']['SO_central_particle'] * BoxSizeHMpc + BoxSizeHMpc/2
    L1_vel   = af['data']['v_com'] * VelZSpace_to_kms
    L1_mass  = af['data']['N'] * ParticleMassHMsun
    af.close()

    L1p_size = L1_pos.shape[0]
    L1v_size = L1_vel.shape[0]
    L1m_size = L1_mass.shape[0]

    assert L1p_size == L1v_size and L1p_size == L1m_size, "Error: L1 arrays have different sizes"

    # Store final arrays of entire version to disk 
    print(f"Compuation finished for {simname} in {time.time()-t0:.2f} seconds. Saving to disk")

    np.save(pos_fname, L1_pos)
    np.save(vel_fname, L1_vel)
    np.save(mass_fname, L1_mass)

    return


def save_L1_pos_vel_mass_ph3000_3500(
        ph_min:     int     = 3000,
        ph_max:     int     = 3500,
        parallel:   bool    = True
        ):
    if parallel:
        # Use parallel processing to save arrays for all phases
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(save_L1_pos_vel_mass_single_version, [ph for ph in range(ph_min, ph_max)])
    else:
        # Save arrays for all phases sequentially
        for v in range(3000, ph_max+1):
            save_L1_pos_vel_mass_single_version(version=v)

# save_L1_pos_vel_mass_single_version(phase=3000)
# save_L1_pos_vel_mass_single_version(phase=3001)

save_L1_pos_vel_mass_ph3000_3500(ph_min=4000, ph_max=5000, parallel=True)

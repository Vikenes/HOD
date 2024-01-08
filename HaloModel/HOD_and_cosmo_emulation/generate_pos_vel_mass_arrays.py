import asdf 
import numpy as np 
import time 
from pathlib import Path

import concurrent.futures
from itertools import repeat

N_files_per_version = 34
ABACUSPATH      = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit") # Location of data 
EMULATION_PATH  = Path(ABACUSPATH / "emulation_files") # Location of data
OUTDIR          = Path("pos_vel_mass_arrays") # name of directory where output arrays are stored 
allowed_files   = [str(i).zfill(3) for i in range(N_files_per_version)]

def get_asdf_filename(
        version:    int = 130,
        file_idx:   int = 0,
        phase:      int = 0,
        ):
    """
    Return the filename of the asdf file for a given version and file number
    """ 
    if file_idx < 0 or file_idx > N_files_per_version:
        print("Error, File must be between 0 and 33")
        return 

    version_str = str(version).zfill(3)
    phase_str   = str(phase).zfill(3)
    simulation  = Path(ABACUSPATH / f"AbacusSummit_base_c{version_str}_ph{phase_str}/halos/z0.250/halo_info/")


    if simulation.exists():
        file          = f"halo_info_{allowed_files[file_idx]}.asdf"
        asdf_filename = Path(simulation / file)
        
    else:
        print(f"Error: directory {simulation} does not exist")
        raise FileNotFoundError

    return asdf_filename


# Get default header info from v130f0 
af_filename = get_asdf_filename()
af          = asdf.open(af_filename)

ParticleMassHMsun  = af['header']['ParticleMassHMsun']
BoxSizeHMpc        = af['header']['BoxSizeHMpc']
VelZSpace_to_kms   = af['header']['VelZSpace_to_kms']


def getMassiveHaloMask(
        version: int = 130, 
        file:    int = 0,
        phase:   int = 0,
        log10Mmin = 12.0,
        ):
    

    """
    Get idx of halos with mass > log10Mmin
    return: mask array of halo indices with mass > log10Mmin
    """
    # log10Mmin = 12.0 #* ParticleMassHMsun

    af_filename     = get_asdf_filename(version, file_idx=file, phase=phase)
    af              = asdf.open(af_filename)
    HaloMassLog10   = np.log10(af['data']['N'] * ParticleMassHMsun)
    MassMask        = HaloMassLog10 > log10Mmin
    af.close()

    return MassMask

def save_L1_pos_vel_mass_single_version(
        version:    int     = 130,
        phase:      int     = 0,
        log10Mmin:  float   = 12.0,
        ):
    """
    Create arrays of L1 data for all files in each simulation version.
    Outputs: 
        pos(N,3): central coordinates 
        vel(N,3): CoM velocities
        mass(N) : Total mass
    Resulting array is saved to disk.  
    """
    assert type(version) == int, "version must be an integer"
    assert type(phase)   == int, "phase must be an integer"

    version_str = str(version).zfill(3)
    phase_str   = str(phase).zfill(3)
    version_dir = Path(f"AbacusSummit_base_c{version_str}_ph{phase_str}")
    
    # Get filename of first file in version, yields error if file does not exist
    af_filename = get_asdf_filename(version=version, file_idx=0, phase=phase)

    # Create output directory if it does not exist
    L1_path = Path(EMULATION_PATH / version_dir / OUTDIR)
    L1_path.mkdir(parents=True, exist_ok=True)

    # Create output filenames
    pos_fname  = Path(L1_path / "L1_pos.npy")
    vel_fname  = Path(L1_path / "L1_vel.npy")
    mass_fname = Path(L1_path / "L1_mass.npy")

    # Check if arrays already exist, if so, skip
    if pos_fname.exists() and vel_fname.exists() and mass_fname.exists():
        print(f"Arrays for {version_dir} already exist, skipping")
        return
    
    # Start computation
    
    print(f"Saving L1 qtys for simulation {version_dir}")
    t0 = time.time()

    # Start with first file in version
    # Abacus splits a simulation into 34 files, each containing a planar slab of width L/34 
    af          = asdf.open(af_filename)
    mask        = getMassiveHaloMask(version=version, file=0, phase=phase, log10Mmin=log10Mmin)

    ### CU = code units
    ### L1_pos_Mpc/h  = L1_pos_CU * BoxSizeHMpc + BoxSizeHMpc/2 
    ### L1_vel_km/s   = L1_vel_CU * VelZSpace_to_kms 
    ### L1_mass_hMsun = L1_N * ParticleMassHMsun
    L1_pos_CU   = af['data']['SO_central_particle'][mask] 
    L1_vel_CU   = af['data']['v_com'][mask]
    L1_N        = af['data']['N'][mask] 
    af.close()

    L1p_size = L1_pos_CU.shape[0]
    L1v_size = L1_vel_CU.shape[0]
    L1N_size = L1_N.shape[0]
    if L1p_size != L1v_size or L1p_size != L1N_size:
        print("Error: L1 arrays have different sizes")
        print(" - L1_pos size: ", L1p_size)
        print(" - L1_vel size: ", L1v_size)
        print(" - L1_N   size: ", L1N_size)
        return

    # Loop over remaining files (planar slabs) in version
    for f in range(1,N_files_per_version):
        af_filename     = get_asdf_filename(version=version, file_idx=f, phase=phase)
        af              = asdf.open(af_filename)
        mask            = getMassiveHaloMask(version=version, file=f, phase=phase)
        L1_pos_CU       = np.concatenate((L1_pos_CU, af['data']['SO_central_particle'][mask])) 
        L1_vel_CU       = np.concatenate((L1_vel_CU, af['data']['v_com'][mask]))
        L1_N            = np.concatenate((L1_N, af['data']['N'][mask] ))
        af.close()
    
    # 
    L1_pos_hMpc     = L1_pos_CU * BoxSizeHMpc + BoxSizeHMpc/2
    L1_vel_kms      = L1_vel_CU * VelZSpace_to_kms
    L1_mass_hMSun   = L1_N * ParticleMassHMsun

    # Store final arrays of entire version to disk 
    print(f"Compuation finished for {version_dir} in {time.time()-t0:.2f} seconds. Saving to disk")

    np.save(pos_fname, L1_pos_hMpc)
    np.save(vel_fname, L1_vel_kms)
    np.save(mass_fname, L1_mass_hMSun)

    return


def save_L1_pos_vel_mass_c000_LCDM_all_phases(parallel=True):
    """
    Stores pos, vel, mass arrays for all phases of the c000 LCDM simulation
    """
    if parallel:
        # Use parallel processing to save arrays for all phases
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(save_L1_pos_vel_mass_single_version, [0 for i in range(25)], [i for i in range(25)])
    else:
        # Save arrays for all phases sequentially
        for i in range(25):
            save_L1_pos_vel_mass_single_version(version=0, phase=i)


def save_L1_pos_vel_mass_version_list(
        v_min:    int  = 130, 
        v_max:    int  = 181, 
        parallel: bool = True
        ):
    if parallel:
        # Use parallel processing to save arrays for all phases
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(save_L1_pos_vel_mass_single_version, [v for v in range(v_min, v_max+1)])
    else:
        # Save arrays for all phases sequentially
        for v in range(v_min, v_max+1):
            save_L1_pos_vel_mass_single_version(version=v)

# save_L1_pos_vel_mass_single_version(version=0, phase=50)
# save_L1_pos_vel_mass_c000_LCDM_all_phases()
# save_L1_pos_vel_mass_version_list(v_min=100, v_max=126, parallel=False)


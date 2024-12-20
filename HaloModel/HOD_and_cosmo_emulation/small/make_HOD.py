import numpy as np 
import pandas as pd
import time
from pathlib import Path
import h5py 

from dq import Cosmology
import hmd 
from hmd.catalogue import HaloCatalogue
from hmd.profiles import FixedCosmologyNFW
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.populate import HODMaker


import warnings
warnings.filterwarnings("ignore", message='Astropy cosmology class contains massive neutrinos, which are not taken into account in Colossus.')

"""
Make HOD catalogues for AbacusSummit_base_c000_ph000-ph024
using the fiducial HOD parameters and cosmology, i.e. one each.

ONLY store the galaxy positions and redshift-space distorted positions.
The catalogues are used to compute the projected corrfunc, wp.

The wp data is then used to compute the covariance matrix.
"""

print()


D13_BASE_PATH       = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
D13_EMULATION_PATH  = f"{D13_BASE_PATH}/emulation_files"
D13_DATA_PATH       = Path(f"{D13_BASE_PATH}/emulation_files/small")
# Common fiducial cosmology for all c000 simulations 
cosmology           = Cosmology.from_custom(run=0, emulator_data_path=f"{D13_EMULATION_PATH}/AbacusSummit_base_c000_ph000")
redshift            = 0.25
boxsize             = 500.0

# Use common fiducial HOD parameters for simulations 
HOD_PARAMS          = pd.read_csv(f"{D13_EMULATION_PATH}/AbacusSummit_base_c000_ph000/HOD_parameters/HOD_parameters_fiducial.csv")
logM_min_array      = np.load("log10Mmin_small.npy")



# Make output files for HOD catalogues
OUTPATH = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data")
outfile = Path(OUTPATH / "halocat_small.hdf5")
if outfile.exists():
    input(f"{outfile} exists. Press enter to overwrite.")

HOD_PROPERTIES_TO_SAVE = ["x", "y", "z", "s_z"]


with h5py.File(outfile, "w") as HOD_file:


    # Loop over all 25 phases of c000 
    phase_range = np.arange(3000, 5000)
    phase_paths = np.array([Path(D13_DATA_PATH / f"AbacusSummit_small_c000_ph{phase}") for phase in phase_range if Path(D13_DATA_PATH / f"AbacusSummit_small_c000_ph{phase}").is_dir()])


    for ii, phase_path in enumerate(phase_paths):
        t0 = time.time()

        # Create group for each simulation
        # simname          = f"AbacusSummit_small_c000_ph{phase}"
        simname = phase_path.name
        HOD_group        = HOD_file.create_group(simname)
                    
        # Load pos, vel and mass of halos with mass > 1e12 h^-1 Msun
        pos  = np.load(phase_path / "pos_vel_mass_arrays/L1_pos.npy")  # shape: (N_halos, 3)
        vel  = np.load(phase_path / "pos_vel_mass_arrays/L1_vel.npy")  # shape: (N_halos, 3)
        mass = np.load(phase_path / "pos_vel_mass_arrays/L1_mass.npy") # shape: (N_halos,)



        # Make halo catalogue. 
        halocat = HaloCatalogue(
            pos,
            vel,
            mass,
            boxsize         = boxsize,
            conc_mass_model = hmd.concentration.diemer15,
            cosmology       = cosmology,
            redshift        = redshift,
            )

        # Populate halos with galaxies
        maker = HODMaker(
            halo_catalogue      = halocat,
            central_occ         = Zheng07Centrals(),
            sat_occ             = Zheng07Sats(),
            satellite_profile   = FixedCosmologyNFW(
                cosmology       = halocat.cosmology,
                redshift        = redshift,
                mdef            = "200m",
                conc_mass_model = "dutton_maccio14",
                sigmaM          = None,
                ),
            galaxy=Galaxy(
            logM_min    = logM_min_array[ii],
            sigma_logM  = HOD_PARAMS['sigma_logM'][0],
            logM1       = HOD_PARAMS['log10M1'][0],
            kappa       = HOD_PARAMS['kappa'][0],
            alpha       = HOD_PARAMS['alpha'][0],
            ),
            )
        maker()

        # Load galaxy catalogue
        galaxy_df           = maker.galaxy_df

        # Add redshift-space distortions to z-coordinate 
        galaxy_df["s_z"] = halocat.apply_redshift_distortion(
            galaxy_df['z'],
            galaxy_df['v_z'],
        )
        
        galaxy_properties = galaxy_df.columns.values.tolist()

        for prop in HOD_PROPERTIES_TO_SAVE:
            HOD_group.create_dataset(
                prop, 
                data = galaxy_df[prop].values,
                dtype= galaxy_df[prop].dtypes
                )
            
        print(f"phase {phase_path.name[-6:]} complete. Duration: {time.time() - t0:.2f} sec.")



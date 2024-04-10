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
Make HOD catalogues for MGGLAMx100
using the fiducial HOD parameters and cosmology, i.e. one each.

ONLY store the galaxy positions and redshift-space distorted positions.
The catalogues are used to compute the projected corrfunc, wp.

The wp data is then used to compute the covariance matrix.
"""

print()


D13_BASE_PATH       = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
D13_EMULATION_PATH  = f"{D13_BASE_PATH}/emulation_files"

# Common fiducial cosmology for all c000 simulations 
cosmology           = Cosmology.from_custom(run=0, emulator_data_path=f"{D13_EMULATION_PATH}/fiducial_data/MGGLAM")
redshift            = 0.25
boxsize             = 1000.0

# Use common fiducial HOD parameters for simulations
HOD_PARAMS          = pd.read_csv(f"{D13_EMULATION_PATH}/AbacusSummit_base_c000_ph000/HOD_parameters/HOD_parameters_fiducial_ng_fixed.csv")

# Get log10Mmin from mean of all MGGLAM sims
from prepare_data import get_log10Mmin_MGGLAM_mean
logM_min    = get_log10Mmin_MGGLAM_mean()

# Make galaxy object
galaxy_fiducial=Galaxy(
    logM_min    = logM_min,
    sigma_logM  = HOD_PARAMS['sigma_logM'][0],
    logM1       = HOD_PARAMS['log10M1'][0],
    kappa       = HOD_PARAMS['kappa'][0],
    alpha       = HOD_PARAMS['alpha'][0]
    )


# Make output files for HOD catalogues
OUTPATH = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data")
outfile = Path(OUTPATH / "halocat_fiducial_MGGLAM_ng_fixed.hdf5")
if outfile.exists():
    input(f"{outfile} exists. Press enter to overwrite.")

HOD_file = h5py.File(outfile, "w")

MGGLAM = h5py.File("/mn/stornext/d8/data/chengzor/MGGLAMx100/GR_halocat_z0.25.hdf5", "r")

HOD_PROPERTIES_TO_SAVE = ["x", "y", "z", "s_z"]


# Loop over all 100 boxes of c000 
for key in MGGLAM.keys():
    t0 = time.time()

    # Create group for each simulation
    HOD_group        = HOD_file.create_group(key)
    MGGLAM_box       = MGGLAM[key]

    # Load pos, vel and mass of halos
    pos  = MGGLAM_box["halo_pos"][:]
    vel  = MGGLAM_box["halo_vel"][:]
    mass = MGGLAM_box["halo_mass"][:]

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
        galaxy=galaxy_fiducial,
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
        
    print(f"{key} complete. Duration: {time.time() - t0:.2f} sec.")

HOD_file.close()
MGGLAM.close()


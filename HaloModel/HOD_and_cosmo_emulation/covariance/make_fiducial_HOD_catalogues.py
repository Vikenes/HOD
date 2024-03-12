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

Used to compute the projected corrfunc, wp, for each one,
which in turn is used to compute the covariance matrix.
"""

print()


D13_BASE_PATH       = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
D13_EMULATION_PATH  = f"{D13_BASE_PATH}/emulation_files"


# Using the fiducial HOD and cosmo parameters for all nodes 
HOD_PARAMS          = pd.read_csv(f"{D13_EMULATION_PATH}/AbacusSummit_base_c000_ph000/HOD_parameters/HOD_parameters_fiducial_ng_fixed.csv")
cosmology           = Cosmology.from_custom(run=0, emulator_data_path=f"{D13_EMULATION_PATH}/AbacusSummit_base_c000_ph000")

OUTPATH = Path(f"{D13_EMULATION_PATH}/fiducial_data/HOD_catalogues")


def make_fiducial_HOD_catalogue_hdf5_files(
    ng_fixed:   bool = True,
    redshift:   float = 0.25,
    boxsize:    float = 2000.0,
    ):

    filename_suffix = ""
    if ng_fixed:
        filename_suffix += "_ng_fixed"
    outfile = Path(OUTPATH / "halocat_fiducial_ng_fixed.hdf5")
    if outfile.exists():
        print(f"{outfile} exists. Skipping.")
        return
    
    fff = h5py.File(outfile, "w")
    
    for phase in range(25):

        t0 = time.time()

        simname         = f"AbacusSummit_base_c000_ph{str(phase).zfill(3)}"
        sim_group       = fff.create_group(simname)
        HOD_DATA_PATH   = f"{D13_EMULATION_PATH}/{simname}"
                    
        # Load pos, vel and mass of halos with mass > 1e12 h^-1 Msun
        pos  = np.load(f"{HOD_DATA_PATH}/pos_vel_mass_arrays/L1_pos.npy")  # shape: (N_halos, 3)
        vel  = np.load(f"{HOD_DATA_PATH}/pos_vel_mass_arrays/L1_vel.npy")  # shape: (N_halos, 3)
        mass = np.load(f"{HOD_DATA_PATH}/pos_vel_mass_arrays/L1_mass.npy") # shape: (N_halos,)


        # Make halo catalogue. 
        halocat = HaloCatalogue(
            pos,
            vel,
            mass,
            boxsize = boxsize,
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
                logM_min    = HOD_PARAMS['log10Mmin'][0],
                sigma_logM  = HOD_PARAMS['sigma_logM'][0],
                logM1       = HOD_PARAMS['log10M1'][0],
                kappa       = HOD_PARAMS['kappa'][0],
                alpha       = HOD_PARAMS['alpha'][0]
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
        
        # Get number of central and satellite galaxies
        galaxy_df_central   = galaxy_df[galaxy_df['galaxy_type'] == 'central']
        galaxy_df_satellite = galaxy_df[galaxy_df['galaxy_type'] == 'satellite']

        # Compute number density of galaxies and store it in hdf5 file
        box_volume = boxsize**3
        ng = len(galaxy_df) / box_volume
        nc = len(galaxy_df_central) / box_volume
        ns = len(galaxy_df_satellite) / box_volume

        sim_group.attrs['ng'] = ng
        sim_group.attrs['nc'] = nc
        sim_group.attrs['ns'] = ns

        # Convert galaxy type from string to int
        galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["central"], 0)
        galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["satellite"], 1)
        galaxy_df.astype({'galaxy_type': int})

        galaxy_properties = galaxy_df.columns.values.tolist()
        
        for prop in galaxy_properties:
            sim_group.create_dataset(
                prop, 
                data = galaxy_df[prop].values,
                dtype= galaxy_df[prop].dtypes
                )
            
        print(f"phase {phase} complete. Duration: {time.time() - t0:.2f} sec.")

    fff.close()


make_fiducial_HOD_catalogue_hdf5_files()
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


D13_EMULATION_PATH  = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files")
OUTPATH             = Path(D13_EMULATION_PATH / "fiducial_data")


HOD_PROPERTIES_TO_SAVE = ["x", "y", "z", "s_z"]


def make_HOD_catalogue_hdf5_file(
        ):


    outfname = f"halocat_fiducial.hdf5"
    OUTFILE = Path(OUTPATH / outfname)

    # Set up cosmology and simulation info 
    # Uses the cosmology.dat file found in the HOD_DATA_PATH
    cosmology = Cosmology.from_custom(run=0, emulator_data_path=f"{D13_EMULATION_PATH}/fiducial_data")
    redshift  = 0.25 
    boxsize   = 2000.0

    # # Make hdf5 file
    with h5py.File(OUTFILE, "w") as fff:
        for phase in range(25):
            simname = f"AbacusSummit_base_c000_ph{str(phase).zfill(3)}"
            SIM_PATH            = Path(D13_EMULATION_PATH / simname)
            HOD_PARAMETERS_PATH = Path(SIM_PATH / "HOD_parameters")
            HALO_ARRAYS_PATH    = Path(SIM_PATH / "pos_vel_mass_arrays") # directory with pos, vel, mass arrays

            # Load pos, vel and mass of halos with mass > 1e12 h^-1 Msun
            pos  = np.load(f"{HALO_ARRAYS_PATH}/L1_pos.npy")  # shape: (N_halos, 3)
            vel  = np.load(f"{HALO_ARRAYS_PATH}/L1_vel.npy")  # shape: (N_halos, 3)
            mass = np.load(f"{HALO_ARRAYS_PATH}/L1_mass.npy") # shape: (N_halos,)

            # Load HOD parameters
            hod_params_fname = f"{HOD_PARAMETERS_PATH}/HOD_parameters_fiducial.csv"
            node_params_df   = pd.read_csv(hod_params_fname)

            # Make halo catalogue. 
            # Note: The halo catalogue is independent of the HOD parameters.
            halocat = HaloCatalogue(
                pos,
                vel,
                mass,
                boxsize,
                conc_mass_model = hmd.concentration.diemer15,
                cosmology       = cosmology,
                redshift        = redshift,
                )


            # Store HOD parameters in hdf5 file
            HOD_group                     = fff.create_group(simname) 
            HOD_group.attrs['log10Mmin']  = node_params_df['log10Mmin'].iloc[0]
            HOD_group.attrs['sigma_logM'] = node_params_df['sigma_logM'].iloc[0]
            HOD_group.attrs['log10M1']    = node_params_df['log10M1'].iloc[0]
            HOD_group.attrs['kappa']      = node_params_df['kappa'].iloc[0]
            HOD_group.attrs['alpha']      = node_params_df['alpha'].iloc[0]
            HOD_group.attrs['log10_ng']   = node_params_df['log10_ng'].iloc[0]

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
                    logM_min    = node_params_df['log10Mmin'].iloc[0],
                    sigma_logM  = node_params_df['sigma_logM'].iloc[0],
                    logM1       = node_params_df['log10M1'].iloc[0],
                    kappa       = node_params_df['kappa'].iloc[0],
                    alpha       = node_params_df['alpha'].iloc[0]
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

            # Compute number density of galaxies and store it in hdf5 file
            box_volume = boxsize**3
            ng = len(galaxy_df) / box_volume
            assert np.abs(ng - 10**node_params_df['log10_ng'].iloc[0]) < 1e-5, f"ng={ng:.5e} != 10^{node_params_df['log10_ng'].iloc[0]:.5f}" 
            HOD_group.attrs['ng'] = ng

            for prop in HOD_PROPERTIES_TO_SAVE:
                HOD_group.create_dataset(
                    prop, 
                    data = galaxy_df[prop].values,
                    dtype= galaxy_df[prop].dtypes
                    )

            print(f"ph{phase}-fiducial complete. ")



make_HOD_catalogue_hdf5_file()
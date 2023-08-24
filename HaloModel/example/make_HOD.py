import numpy as np 
import h5py 
import time 
from dq import Cosmology
import hmd
from hmd.catalogue import ParticleCatalogue, HaloCatalogue, GalaxyCatalogue
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW
from hmd.populate import HODMaker
from pathlib import Path

HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData"

import warnings
warnings.filterwarnings("ignore")

OUTFILE = Path(f"{HOD_DATA_PATH}/HOD_abacus_cat_z0.25.hdf5")

with h5py.File(OUTFILE, "w") as fff:
    redshift = 0.25 
    boxsize  = 2000.0
    fff.attrs["boxsize"]  = boxsize
    fff.attrs["central_galaxy"] = 0 
    fff.attrs["satellite_galaxy"] = 1
    # f.attrs["redshift"] = redshift
    
    file_halocat  = h5py.File(f"{HOD_DATA_PATH}/abacus_halo_cat_z0.25.hdf5", 'r')
    N_simulations = len(file_halocat.keys())

    for run in range(N_simulations):
        version_idx = int(run + 130)
        print(f"Running version{version_idx}...", end=" ")
        t0 = time.time()

        HOD_group = fff.create_group(f"version{version_idx}") 

        cosmology = Cosmology.from_custom(run=run, emulator_data_path=HOD_DATA_PATH)
        
        HOD_group.attrs["H0"]   = float(cosmology.H0.value)
        HOD_group.attrs["Om0"]  = cosmology.Om0
        HOD_group.attrs["Ode0"] = cosmology.Ode0
        HOD_group.attrs["w0"]   = cosmology.w0
        HOD_group.attrs["wc0"]  = cosmology.wc0
        HOD_group.attrs["Ob0"]  = cosmology.Ob0
        HOD_group.attrs["Neff"] = cosmology.Neff
        HOD_group.attrs["lnAs"] = cosmology.lnAs
        HOD_group.attrs["n_s"]  = cosmology.n_s


        pos  = file_halocat[f"version{version_idx}"]["halo_pos"][...] # shape: (N_halos, 3)
        vel  = file_halocat[f"version{version_idx}"]["halo_vel"][...]
        mass = file_halocat[f"version{version_idx}"]["halo_mass"][...]


        halocat = HaloCatalogue(
            pos,
            vel,
            mass,
            boxsize,
            conc_mass_model=hmd.concentration.diemer15,
            cosmology=cosmology,
            redshift=redshift,
            # mdef: str = "200m",
        )


        maker = HODMaker(
            halo_catalogue=halocat,
            central_occ=Zheng07Centrals(),
            sat_occ=Zheng07Sats(),
            satellite_profile=FixedCosmologyNFW(
                cosmology=halocat.cosmology,
                redshift=redshift,
                mdef="200m",
                conc_mass_model="dutton_maccio14",
                sigmaM=None,
            ),
            galaxy=Galaxy(
                logM_min = 13.62,
                sigma_logM = 0.6915,
                kappa = 0.51,
                logM1 = 14.42,
                alpha = 0.9168,
                # M_cut = 10 ** 12.26,
                # M_sat = 10 ** 14.87,
                # concentration_bias = 1.0,
                # v_bias_centrals = 0.0,
                # v_bias_satellites = 1.0,
                # B_cen = 0.,
                # B_sat = 0.,
            ),
        )

        maker()
        galaxy_df = maker.galaxy_df
        galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["central"], 0)
        galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["satellite"], 1)
        galaxy_df.astype({'galaxy_type': int})
        galaxy_properties = galaxy_df.columns.values.tolist()
        for prop in galaxy_properties:
            HOD_group.create_dataset(
                prop, 
                data=galaxy_df[prop].values,
                dtype=galaxy_df[prop].dtypes
                )
        print(f"Finised, took {time.time() - t0:.2f} seconds.")

    file_halocat.close()
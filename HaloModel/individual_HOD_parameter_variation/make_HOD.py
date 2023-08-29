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
import pandas as pd

# import warnings
# warnings.filterwarnings("ignore")

HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
HALO_ARRAYS_PATH = f"{HOD_DATA_PATH}/version0"

OUTFILEPATH = f"{HOD_DATA_PATH}/individual_HOD_parameter_variation"

dataset_names = [
    'sigma_logM', 
    'log10M1', 
    'kappa',
    'alpha']

def make_hdf5_files(ng_fixed=False):

    print("Making hdf5 files...")

    for param in dataset_names:
        print(f"Computing for param={param}")
        outfname = f"halocat_vary_{param}"
        OUTFILE = Path((f"{OUTFILEPATH}/{outfname}.hdf5"))
        if OUTFILE.exists():
            print(f"File {OUTFILE} already exists, skipping...")
            continue

        fff = h5py.File(f"{OUTFILEPATH}/{outfname}.hdf5", "w")

        hod_params_fname = f"{OUTFILEPATH}/HOD_parameters_vary_{param}.csv"

        node_params_df = pd.read_csv(hod_params_fname)

        cosmology = Cosmology.from_custom(run=0, emulator_data_path=HOD_DATA_PATH)

        redshift = 0.25 
        boxsize  = 2000.0
        fff.attrs["boxsize"]  = boxsize
        fff.attrs["central_galaxy"] = 0 
        fff.attrs["satellite_galaxy"] = 1
        fff.attrs["H0"]   = float(cosmology.H0.value)
        fff.attrs["Om0"]  = cosmology.Om0
        fff.attrs["Ode0"] = cosmology.Ode0
        fff.attrs["w0"]   = cosmology.w0
        fff.attrs["wc0"]  = cosmology.wc0
        fff.attrs["Ob0"]  = cosmology.Ob0
        fff.attrs["Neff"] = cosmology.Neff
        fff.attrs["lnAs"] = cosmology.lnAs
        fff.attrs["n_s"]  = cosmology.n_s
        # f.attrs["redshift"] = redshift

        pos  = np.load(f"{HALO_ARRAYS_PATH}/L1_pos.npy") # shape: (N_halos, 3)
        vel  = np.load(f"{HALO_ARRAYS_PATH}/L1_vel.npy") #file_halocat[f"version{version_idx}"]["halo_vel"][...]
        mass = np.load(f"{HALO_ARRAYS_PATH}/L1_mass.npy")#file_halocat[f"version{version_idx}"]["halo_mass"][...]
        
        
        halocat = HaloCatalogue(
            pos,
            vel,
            mass,
            boxsize,
            conc_mass_model=hmd.concentration.diemer15,
            cosmology=cosmology,
            redshift=redshift,
            )

        for node_idx in range(len(node_params_df)):
            print(f"Running node{node_idx}...", end=" ")
            t0 = time.time()

            HOD_group = fff.create_group(f"node{node_idx}") 

            HOD_group.attrs['log10Mmin']     = node_params_df['log10Mmin'].iloc[node_idx]
            HOD_group.attrs['sigma_logM']    = node_params_df['sigma_logM'].iloc[node_idx]
            HOD_group.attrs['log10M1']       = node_params_df['log10M1'].iloc[node_idx]
            HOD_group.attrs['kappa']         = node_params_df['kappa'].iloc[node_idx]
            HOD_group.attrs['alpha']         = node_params_df['alpha'].iloc[node_idx]

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
                    logM_min    = node_params_df['log10Mmin'].iloc[node_idx],
                    sigma_logM  = node_params_df['sigma_logM'].iloc[node_idx],
                    logM1       = node_params_df['log10M1'].iloc[node_idx],
                    kappa       = node_params_df['kappa'].iloc[node_idx],
                    alpha       = node_params_df['alpha'].iloc[node_idx],
            
                ),
            )

            maker()
            galaxy_df = maker.galaxy_df
            galaxy_df_central = galaxy_df[galaxy_df['galaxy_type'] == 'central']
            galaxy_df_satellite = galaxy_df[galaxy_df['galaxy_type'] == 'satellite']
            Ng = len(galaxy_df)
            Nc = len(galaxy_df_central)
            Ns = len(galaxy_df_satellite)
            ng = Ng / (boxsize**3)
            nc = Nc / (boxsize**3)
            ns = Ns / (boxsize**3)
            HOD_group.attrs['ng'] = ng
            HOD_group.attrs['nc'] = nc
            HOD_group.attrs['ns'] = ns

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

        fff.close()


make_hdf5_files(ng_fixed=True)
# make_HOD_fiducial()
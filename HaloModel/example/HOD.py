import numpy as np 
import h5py 
from dq import Cosmology
import hmd
from hmd.catalogue import ParticleCatalogue, HaloCatalogue, GalaxyCatalogue
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW
from hmd.populate import HODMaker
from pathlib import Path

D13_BASE_PATH           = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
simname             = f"AbacusSummit_base_c000_ph000"
HOD_DATA_PATH       = f"{D13_BASE_PATH}/emulation_files/{simname}"
HALO_ARRAYS_PATH    = f"{HOD_DATA_PATH}/pos_vel_mass_arrays"
# HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData"

# halocat_fname = Path(f"{HOD_DATA_PATH}/abacus_halo_cat_z0.25.hdf5")
# file_halocat = h5py.File(
#     halocat_fname, 
#     'r',
# )
# N_simulations = len(file_halocat.keys())

boxsize = 2000.0
redshift = 0.25

# run = 32 
# version_index = int(run + 130)

pos  = np.load(f"{HALO_ARRAYS_PATH}/L1_pos.npy")  # shape: (N_halos, 3)
vel  = np.load(f"{HALO_ARRAYS_PATH}/L1_vel.npy")  # shape: (N_halos, 3)
mass = np.load(f"{HALO_ARRAYS_PATH}/L1_mass.npy") # shape: (N_halos,)
# pos  = file_halocat[f"version{version_index}"]["halo_pos"][...] # shape: (N_halos, 3)
# vel  = file_halocat[f"version{version_index}"]["halo_vel"][...]
# mass = file_halocat[f"version{version_index}"]["halo_mass"][...]
# for run in range(52):
    # print(f"run={run}:", end=" ")
cosmology = Cosmology.from_custom(run=0, emulator_data_path=HOD_DATA_PATH)
# exit()

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
halo_df = halocat.to_frame()
conc = hmd.concentration.diemer15(
                prim_haloprop=mass,
                cosmology=cosmology,
                redshift=redshift,
                sigma8=cosmology.get_sigma8()
                if hasattr(cosmology, "get_sigma8")
                else None,
            )
print(halo_df["concentration"])
print()
print(conc)
exit()

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

# galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].astype(str)
galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["central"], 0)
galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["satellite"], 1)
# print(type(galaxy_df["galaxy_type"]))

# galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].astype(str)
galaxy_df.astype({'galaxy_type': int})


# galaxy_df.astype({'galaxy_type': str})


galaxy_properties = galaxy_df.columns.values.tolist()
# print(type(galaxy_df["galaxy_type"]))
print(galaxy_df["galaxy_type"].dtypes)
exit()
for prop in galaxy_properties:
    if prop != 'galaxy_type':
        continue

    print(f"prop={prop}"),
    print()
    print(galaxy_df[prop])
    print("---")
    print(galaxy_df[prop].values) 
    # data=galaxy_df[prop].values,
    # dtype=galaxy_df[prop].dtypes
        # )
    print('--')
    print(galaxy_df[prop].dtypes)
    exit()


exit()
fname = Path(f"./v{version_index}_HOD.csv") 
if fname.exists():
    print(f"File {fname} already exists, aborting...")
    exit()

maker.galaxy_df.to_csv(
    f"./v{version_index}_HOD.csv",
)

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
import sys 

from numdens_HOD import estimate_log10Mmin_from_gal_num_density

# import warnings
# warnings.filterwarnings("ignore")

DATA_PATH           = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
HALO_ARRAYS_PATH    = f"{DATA_PATH}/version0"
HOD_DATA_PATH       = f"{DATA_PATH}/TPCF_emulation"
HOD_PARAMETERS_PATH = f"{HOD_DATA_PATH}/HOD_parameters"
OUTFILEPATH         = f"{HOD_DATA_PATH}/HOD_catalogues"

dataset_names = ['train', 'test', 'val']

def make_hdf5_files(ng_fixed=True, pos_only=False):

    print("Making hdf5 files...")

    for flag in dataset_names:
        
        print(f"Computing for flag={flag}")
        filename_suffix = f"{flag}"
        if ng_fixed:
            filename_suffix += "_ng_fixed"
        
        outfname = f"halocat_{filename_suffix}"
        if pos_only:
            outfname += "_pos"

        OUTFILE = Path((f"{OUTFILEPATH}/{outfname}.hdf5"))
        if OUTFILE.exists():
            print(f"File {OUTFILE} already exists, skipping...")
            continue
        
        # Make hdf5 file
        fff = h5py.File(f"{OUTFILEPATH}/{outfname}.hdf5", "w")

        # Set up cosmology and simulation info 
        cosmology = Cosmology.from_custom(run=0, emulator_data_path=DATA_PATH)
        redshift  = 0.25 
        boxsize   = 2000.0

        # Store cosmology and simulation info in hdf5 file
        fff.attrs["boxsize"]            = boxsize
        fff.attrs["central_galaxy"]     = 0
        fff.attrs["satellite_galaxy"]   = 1
        fff.attrs["H0"]                 = float(cosmology.H0.value)
        fff.attrs["Om0"]                = cosmology.Om0
        fff.attrs["Ode0"]               = cosmology.Ode0
        fff.attrs["w0"]                 = cosmology.w0
        fff.attrs["wc0"]                = cosmology.wc0
        fff.attrs["Ob0"]                = cosmology.Ob0
        fff.attrs["Neff"]               = cosmology.Neff
        fff.attrs["lnAs"]               = cosmology.lnAs
        fff.attrs["n_s"]                = cosmology.n_s

        # Load pos, vel and mass of halos with mass > 1e12 h^-1 Msun
        pos  = np.load(f"{HALO_ARRAYS_PATH}/L1_pos.npy")  # shape: (N_halos, 3)
        vel  = np.load(f"{HALO_ARRAYS_PATH}/L1_vel.npy")  # shape: (N_halos, 3)
        mass = np.load(f"{HALO_ARRAYS_PATH}/L1_mass.npy") # shape: (N_halos,)

        # Load HOD parameters
        hod_params_fname = f"{HOD_PARAMETERS_PATH}/HOD_parameters_{filename_suffix}.csv"
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

        # Loop over HOD parameters 
        # Populate halos with galaxies for each HOD parameter set 
        for node_idx in range(len(node_params_df)):

            print(f"Running node{node_idx}...", end=" ")
            t0 = time.time()

            # Store HOD parameters in hdf5 file
            HOD_group = fff.create_group(f"node{node_idx}") 
            HOD_group.attrs['log10Mmin']     = node_params_df['log10Mmin'].iloc[node_idx]
            HOD_group.attrs['sigma_logM']    = node_params_df['sigma_logM'].iloc[node_idx]
            HOD_group.attrs['log10M1']       = node_params_df['log10M1'].iloc[node_idx]
            HOD_group.attrs['kappa']         = node_params_df['kappa'].iloc[node_idx]
            HOD_group.attrs['alpha']         = node_params_df['alpha'].iloc[node_idx]

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
                    logM_min    = node_params_df['log10Mmin'].iloc[node_idx],
                    sigma_logM  = node_params_df['sigma_logM'].iloc[node_idx],
                    logM1       = node_params_df['log10M1'].iloc[node_idx],
                    kappa       = node_params_df['kappa'].iloc[node_idx],
                    alpha       = node_params_df['alpha'].iloc[node_idx]
                    ),
                )
            maker()

            # Load galaxy catalogue
            galaxy_df           = maker.galaxy_df

            # Get number of central and satellite galaxies
            # Compute number density of galaxies and store it in hdf5 file
            galaxy_df_central   = galaxy_df[galaxy_df['galaxy_type'] == 'central']
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

            # Convert galaxy type from string to int
            galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["central"], 0)
            galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["satellite"], 1)
            galaxy_df.astype({'galaxy_type': int})

            # Store galaxy catalogue in hdf5 file
            # galaxy_properties:
            #  - host_mass, host_radius, host_concentration
            #  - x, y, z, v_x, v_y, v_z
            #  - galaxy_type
            #  - host_centric_distance, host_id 


            galaxy_properties = galaxy_df.columns.values.tolist()
            pos_keys          = ['x', 'y', 'z']
            
            for prop in galaxy_properties:
                if prop not in pos_keys and pos_only:
                    continue
                HOD_group.create_dataset(
                    prop, 
                    data = galaxy_df[prop].values,
                    dtype= galaxy_df[prop].dtypes
                    )
                
            print(f"Finished, took {time.time() - t0:.2f} seconds.")

        fff.close()

def make_HOD_fiducial_cosmology(
        ng_fixed=True,
        pos_only=False,
        ):
    """
    Generate HOD parameters for the fiducial model.
    Only used to test implementation in the beginning, and as a reference point for parameter values. 
    """

    if ng_fixed:
        fname_suffix = "_ng_fixed"
    else:
        fname_suffix = ""

    outfname = f"halocat_fiducial_cosmology{fname_suffix}.hdf5"
    OUTFILE  = f"{OUTFILEPATH}/{outfname}"

    if Path((OUTFILE)).exists():
            print(f"File {OUTFILEPATH}/halocat_fiducial.hdf5 already exists, skipping...")
            return 

    sigma_logM  = 0.6915
    log10M1     = 14.42 # h^-1 Msun
    kappa       = 0.51
    alpha       = 0.9168
    if ng_fixed:
        log10Mmin = estimate_log10Mmin_from_gal_num_density(
            sigma_logM_array = sigma_logM, 
            log10M1_array    = log10M1, 
            kappa_array      = kappa, 
            alpha_array      = alpha,
            ng_desired       = 2.174e-4)[0]
    else:
        log10Mmin   = 13.62 # h^-1 Msun

    
    node_params = [log10Mmin, sigma_logM, log10M1, kappa, alpha]

    cosmology = Cosmology.from_custom(run=0, emulator_data_path=DATA_PATH)

    redshift = 0.25 
    boxsize  = 2000.0
    
    pos  = np.load(f"{HALO_ARRAYS_PATH}/L1_pos.npy") 
    vel  = np.load(f"{HALO_ARRAYS_PATH}/L1_vel.npy")
    mass = np.load(f"{HALO_ARRAYS_PATH}/L1_mass.npy")

    halocat = HaloCatalogue(
        pos,
        vel,
        mass,
        boxsize,
        conc_mass_model=hmd.concentration.diemer15,
        cosmology=cosmology,
        redshift=redshift,
        )

    print(f"Running Fiducial...")
    t0 = time.time()

    # HOD_group = fff.create_group(f"node{node_idx}") 
    fff = h5py.File(OUTFILE, "w")


    fff.attrs['log10Mmin']     = node_params[0]
    fff.attrs['sigma_logM']    = node_params[1]
    fff.attrs['log10M1']       = node_params[2]
    fff.attrs['kappa']         = node_params[3]
    fff.attrs['alpha']         = node_params[4]

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
            logM_min    = node_params[0],
            sigma_logM  = node_params[1],
            logM1       = node_params[2],
            kappa       = node_params[3],
            alpha       = node_params[4],
    
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
    fff.attrs['ng'] = ng
    fff.attrs['nc'] = nc
    fff.attrs['ns'] = ns

    galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["central"], 0)
    galaxy_df['galaxy_type'] = galaxy_df['galaxy_type'].replace(["satellite"], 1)
    galaxy_df.astype({'galaxy_type': int})
    galaxy_properties = galaxy_df.columns.values.tolist()
    pos_keys = ['x', 'y', 'z']
    
    for prop in galaxy_properties:
        if prop not in pos_keys and pos_only:
            continue

        fff.create_dataset(
            prop, 
            data=galaxy_df[prop].values,
            dtype=galaxy_df[prop].dtypes
            )
    print(f"Finised, took {time.time() - t0:.2f} seconds.")

    fff.close()


# make_HOD_fiducial_cosmology(ng_fixed=True, pos_only=False)
import numpy as np 
import pandas as pd
import time
from pathlib import Path
import h5py 
from pycorr import TwoPointCorrelationFunction

from numdens_HOD import estimate_log10Mmin_from_gal_num_density

from dq import Cosmology
import hmd 
from hmd.catalogue import HaloCatalogue
from hmd.profiles import FixedCosmologyNFW
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.populate import HODMaker

D13_BASE_PATH       = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"
D13_EMULATION_PATH  = f"{D13_BASE_PATH}/emulation_files"
SIMNAME             = "AbacusSummit_base_c000_ph000"
HOD_DATA_PATH       = f"{D13_EMULATION_PATH}/{SIMNAME}"

HOD_PARAMETERS_PATH = Path(f"{D13_EMULATION_PATH}/{SIMNAME}/HOD_parameters")
HOD_CATALOGUE_PATH  = f"{HOD_DATA_PATH}/HOD_catalogues"
HALO_ARRAYS_PATH    = f"{HOD_DATA_PATH}/pos_vel_mass_arrays" # directory with pos, vel, mass arrays
TPCF_DATA_PATH      = f"{HOD_DATA_PATH}/TPCF_data"

def make_csv_file(
        fix_ng:     bool  = True,
        ng_desired: float = 2.174e-4, # h^3 Mpc^-3
        ):
    """
    Create parameter datasets with LHS sampler.
    generates   num_train nodes for training, 
                num_test nodes for testing,
                num_val nodes for validation.

    If fix_ng=True, log10Mmin is estimated such that the galaxy number density becomes ng_desired.
    """

    print("Making csv file for fiducial HOD parameters.")
    ### FIDUCIAL VALUES. From https://arxiv.org/pdf/2208.05218.pdf, Table 3  ###
    sigma_logM  = 0.6915 
    log10M1     = 14.42 # h^-1 Msun
    kappa       = 0.51 
    alpha       = 0.9168  


    if not fix_ng:
        # Vary log10Mmin by +/- 10% around fiducial value
        log10Mmin           = 13.62 # h^-1 Msun

    # Create parameter files. 

    # Check if simulation data exists, if not, raise error
    # Prevents creation of files for non-existing simulations!
    HOD_PARAMETERS_PATH.mkdir(parents=False, exist_ok=True)

    fname = f"HOD_parameters_fiducial"
    if fix_ng:
        fname += "_ng_fixed" 
    fname   = Path(f"{fname}.csv")
    outfile = Path(HOD_PARAMETERS_PATH / fname)
    if outfile.exists():
        print(f"File {outfile} already exists. Skipping.")
        return 
    
    # Sample parameters
    if fix_ng:
        start       = time.time()
        log10Mmin   = estimate_log10Mmin_from_gal_num_density(
            sigma_logM_array    = [sigma_logM],
            log10M1_array       = [log10M1],
            kappa_array         = [kappa],
            alpha_array         = [alpha],
            ng_desired          = ng_desired,
            test                = False,
            )[0]
        
        print(f"Estimating log10Mmin for fiducial dataset took {time.time() - start:.2f} seconds.")
    if type(log10Mmin) == np.ndarray:
        log10Mmin = log10Mmin[0]
    # Save parameters to csv file
    df = pd.DataFrame({
        'log10Mmin'     : [log10Mmin],
        'sigma_logM'    : [sigma_logM],
        'log10M1'       : [log10M1],
        'kappa'         : [kappa],
        'alpha'         : [alpha],
    })
    
    # Save to csv file
    df.to_csv(
        outfile,
        index=False
    )
    print(f"Saved {outfile}")



def make_HOD_catalogue_hdf5_file(
        ng_fixed:   bool = True
        ):


    Path(HOD_CATALOGUE_PATH).mkdir(parents=False, exist_ok=True)

    print(f"Making fiducial HOD hdf5 file for {SIMNAME}, ...")

    filename_suffix = "fiducial"
    if ng_fixed:
        filename_suffix += "_ng_fixed"
    
    outfname = f"halocat_{filename_suffix}.hdf5"
    OUTFILE = Path((f"{HOD_CATALOGUE_PATH}/{outfname}"))
    if OUTFILE.exists():
        # Prevent overwriting to save time 
        print(f"File {OUTFILE} already exists, skipping...")
        return 

    

    # Set up cosmology and simulation info 
    # Uses the cosmology.dat file found in the HOD_DATA_PATH
    cosmology = Cosmology.from_custom(run=0, emulator_data_path=HOD_DATA_PATH)
    redshift  = 0.25 
    boxsize   = 2000.0

    # # Make hdf5 file
    fff = h5py.File(f"{HOD_CATALOGUE_PATH}/{outfname}", "w")
    
    # # Store cosmology and simulation info in hdf5 file
    fff.attrs["boxsize"]            = boxsize
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
    t0 = time.time()
    for node_idx in range(len(node_params_df)):

        # Store HOD parameters in hdf5 file
        HOD_group                     = fff.create_group(f"node{node_idx}") 
        HOD_group.attrs['log10Mmin']  = node_params_df['log10Mmin'].iloc[node_idx]
        HOD_group.attrs['sigma_logM'] = node_params_df['sigma_logM'].iloc[node_idx]
        HOD_group.attrs['log10M1']    = node_params_df['log10M1'].iloc[node_idx]
        HOD_group.attrs['kappa']      = node_params_df['kappa'].iloc[node_idx]
        HOD_group.attrs['alpha']      = node_params_df['alpha'].iloc[node_idx]

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
        galaxy_df_central   = galaxy_df[galaxy_df['galaxy_type'] == 'central']
        galaxy_df_satellite = galaxy_df[galaxy_df['galaxy_type'] == 'satellite']

        # Compute number density of galaxies and store it in hdf5 file
        box_volume = boxsize**3
        ng = len(galaxy_df) / box_volume
        nc = len(galaxy_df_central) / box_volume
        ns = len(galaxy_df_satellite) / box_volume

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
        
        for prop in galaxy_properties:
            HOD_group.create_dataset(
                prop, 
                data = galaxy_df[prop].values,
                dtype= galaxy_df[prop].dtypes
                )

    fff.close()
    print(f"{SIMNAME}-fiducial complete. Duration: {time.time() - t0:.2f} sec.")


def make_TPCF_data_from_HOD_catalogue(threads=40):
    # Create output directory if it doesn't exist
    Path(TPCF_DATA_PATH).mkdir(parents=False, exist_ok=True)
    fname       = f"TPCF_fiducial_ng_fixed.hdf5"
    OUTFILE     = f"{TPCF_DATA_PATH}/{fname}"
    

    if Path(OUTFILE).exists():
        print(f"File {OUTFILE} already exists, skipping...")
        return 

    halo_file   = h5py.File(f"{HOD_CATALOGUE_PATH}/halocat_fiducial_ng_fixed.hdf5", "r")
    N_nodes     = len(halo_file.keys()) # Number of parameter samples used to make catalogue

    r_bin_edges = np.concatenate((
        np.logspace(np.log10(0.01), np.log10(5), 40, endpoint=False),
        np.linspace(5.0, 150.0, 75)
    ))

    print(f"Computing {fname}...")
    t0 = time.time()
    fff = h5py.File(OUTFILE, "w")
    # Compute TPCF for each node
    for node_idx in range(N_nodes):
        # Load galaxy positions for node from halo catalogue
        HOD_node_catalogue = halo_file[f"node{node_idx}"]
        x = np.array(HOD_node_catalogue['x'][:])
        y = np.array(HOD_node_catalogue['y'][:])
        z = np.array(HOD_node_catalogue['z'][:])
        
        # Compute TPCF 
        result = TwoPointCorrelationFunction(
                mode            = 's',
                edges           = r_bin_edges,
                data_positions1 = np.array([x, y, z]),
                boxsize         = halo_file.attrs['boxsize'],
                engine          = "corrfunc",
                nthreads        = 20,
                )
        r, xi = result(return_sep=True)
       
        # Save TPCF to file
        node_group = fff.create_group(f'node{node_idx}')
        node_group.create_dataset("r",  data=r)
        node_group.create_dataset("xi", data=xi)

    print(f"Done. Took {time.time() - t0:.2f} s")
    fff.close()
    halo_file.close()

# make_csv_file()
# make_HOD_catalogue_hdf5_file()

make_TPCF_data_from_HOD_catalogue()
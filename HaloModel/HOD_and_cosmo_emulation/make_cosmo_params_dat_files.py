import asdf 
import numpy as np
import pandas as pd
from pathlib import Path

"""
Create a .dat table of cosmological parameters for the AbacusSummit simulations
Used to define the cosmology needed to create the HOD catalogues.
"""

ABACUSPATH = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit")
EMULATION_PATH = Path(ABACUSPATH / "emulation_files")
HODPATH = Path("/mn/stornext/d5/data/vetleav/HOD_AbacusData")
C000PATH = Path(HODPATH / "c000_LCDM_simulation") 
LOCALPATH = Path(".")

N_versions = 52
N_files_per_version = 34 
# allowed_files  = [str(i).zfill(3) for i in range(N_files_per_version)]

def get_asdf_version_header(
        version: int = 130,
        phase:   int = 0,
        ):
    """
    Return the filename of the asdf file for a given version and file number
    """
    version_str = str(version).zfill(3)
    phase_str   = str(phase).zfill(3)
    simulation  = Path(ABACUSPATH / f"AbacusSummit_base_c{version_str}_ph{phase_str}/halos/z0.250/halo_info/")


    if simulation.exists():
        asdf_filename = Path(simulation / "halo_info_000.asdf")
        
    else:
        print(f"Error: directory {simulation} does not exist")
        return
    af = asdf.open(asdf_filename)#['header']
    header = af['header']
    af.close()
    return header 

def get_sim_params_from_csv_table(version=130):
    cosmologies_file    = Path(ABACUSPATH / "cosmologies.csv")
    cosmologies         = pd.read_csv(cosmologies_file)

    ## Get the index of the version
    ## The csv contains much whitespace, so we extract the version number from the column names
    colnames            = cosmologies.columns # Column names
    sim_names_colname   = colnames[0]   # First column is the simulation names
    sim_names           = cosmologies[sim_names_colname].values # List of simulation names: abacus_cosm_vvv
    sim_names_versions  = [int(sim_name[11:]) for sim_name in sim_names] # Version number of each simulation
    idx                 = np.where(np.array(sim_names_versions) == version)[0][0] # Index of the version we want
    
    sim_params          = cosmologies.iloc[idx]#[2:] # LCDM parameters

    version_csv = int(sim_params.iloc[0].strip()[-3:])
    if version_csv != version:
        print(f"Error: version {version} does not match version in csv file {version_csv}")
        exit()
    
    return sim_params

def get_As_from_csv_table(version=130):
    sim_params = get_sim_params_from_csv_table(version=version)
    As = sim_params.iloc[5]
    return As

def get_sigma8m_from_csv_table(version=130):
    sim_params = get_sim_params_from_csv_table(version=version)
    sigma8m = sim_params.iloc[13]
    return sigma8m


def save_cosmo_parameters_c000_all_phases(version=0):
    """
    All phases have the same parameters. 
    Get the parameters from the first phase.
    Store copies of the file in the directory of each phase.
    """
    header      = get_asdf_version_header(version=version)
    redshift    = header['Redshift']
    wb          = header['omega_b']
    wc          = header['omega_cdm']
    Ol          = header['Omega_DE']
    As          = get_As_from_csv_table(version=version)
    lnAs        = np.log(As * 1.0e10)
    ns          = header['n_s']
    w           = header['w']
    sigma8      = get_sigma8m_from_csv_table(version=version) 
    Om          = header['Omega_M']
    h           = header['H0'] / 100.0     
    N_eff       = header['N_ncdm'] + header['N_ur']

    df = pd.DataFrame({
        'version'   : version,
        'redshift'  : redshift,
        'wb'        : wb,
        'wc'        : wc,
        'Ol'        : Ol,
        'lnAs'      : lnAs,
        'ns'        : ns,
        'w'         : w,
        'sigma8'    : sigma8,
        'Om'        : Om,
        'h'         : h,
        'N_eff'     : N_eff
    }, index=[0])

    # Store df in each phase directory
    filename = Path(f"cosmological_parameters.dat")
    print(f"Saving {filename} for all phases.")
    for ph in range(0, 25):
        phase_str = str(ph).zfill(3)
        VERSIONPATH = Path(EMULATION_PATH / f"AbacusSummit_base_c000_ph{phase_str}")
        VERSIONPATH.mkdir(parents=True, exist_ok=True)
        filename = Path("cosmological_parameters.dat")
        paramfile = Path(VERSIONPATH / filename)
        if paramfile.exists():
            print(f"Error: file {filename} already exists. Skipping.")
            continue

        df.to_csv(paramfile, index=False, sep=" ")

    print("Finished")


def save_cosmo_parameters_all_versions():
    filename = Path("cosmological_parameters.dat")
    versions = np.arange(0,4)
    # versions = np.concatenate((versions, np.arange(100, 127)))
    versions = np.concatenate((versions, np.arange(130,182)))
    print("Saving cosmological parameters for versions: ", end="")
    for ver in versions:
        print(f"{ver}", end=", ")
        version_str = str(ver).zfill(3)
        VERSIONPATH = Path(EMULATION_PATH / f"AbacusSummit_base_c{version_str}_ph000")
        VERSIONPATH.mkdir(parents=True, exist_ok=True)
        paramfile = Path(VERSIONPATH / filename)
        if paramfile.exists():
            print(f"Error: file {filename} already exists. Skipping.")
            continue
        
        header      = get_asdf_version_header(version=ver)
        redshift    = header['Redshift']
        wb          = header['omega_b']
        wc          = header['omega_cdm']
        Ol          = header['Omega_DE']
        As          = get_As_from_csv_table(version=ver)
        lnAs        = np.log(As * 1.0e10)
        ns          = header['n_s']
        w           = header['w']
        sigma8      = get_sigma8m_from_csv_table(version=ver) 
        Om          = header['Omega_M']
        h           = header['H0'] / 100.0     
        N_eff       = header['N_ncdm'] + header['N_ur']

        df = pd.DataFrame({
            'version'   : ver,
            'redshift'  : redshift,
            'wb'        : wb,
            'wc'        : wc,
            'Ol'        : Ol,
            'lnAs'      : lnAs,
            'ns'        : ns,
            'w'         : w,
            'sigma8'    : sigma8,
            'Om'        : Om,
            'h'         : h,
            'N_eff'     : N_eff
        }, index=[0])
        df.to_csv(paramfile, index=False, sep=" ")

    print("Finished")

# save_cosmo_parameters_c000_all_phases()
save_cosmo_parameters_all_versions()
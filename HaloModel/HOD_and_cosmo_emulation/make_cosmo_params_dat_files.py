import asdf 
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

"""
Create a .dat table of cosmological parameters for the AbacusSummit simulations
Used to define the cosmology needed to create the HOD catalogues.
"""

D13_BASE_PATH       = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit")
D13_EMULATION_PATH  = Path(D13_BASE_PATH / "emulation_files")
OUTFILENAME = filename = Path(f"cosmological_parameters.dat") # Same filename for all simulations

csv_column_names = ['omega_b', 'omega_cdm', 'h', 
                    'A_s', 'n_s', 'alpha_s', 
                    'N_ur', 'N_ncdm', 'omega_ncdm', 
                    'w0_fld', 'wa_fld', 'sigma8_m', 
                    'sigma8_cb']

def get_asdf_version_header(
        version: int = 130,
        phase:   int = 0,
        ):
    """
    Return the header of the asdf file for a given version and file number
    """
    simname     = Path(f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}")
    simulation  = Path(D13_BASE_PATH / simname / "halos/z0.250/halo_info/")


    if not simulation.exists():        
        print(f"Error: directory {simulation} does not exist")
        return
    
    # Same header for all files, load the first one 
    asdf_filename = Path(simulation / "halo_info_000.asdf")
    
    af = asdf.open(asdf_filename)
    header = af['header']
    af.close()
    return header 

def get_sim_params_from_csv_table(
        version:    int = 130,
        param_name: Optional[str] = None,):
    
    cosmologies_file    = Path(D13_BASE_PATH / "cosmologies.csv")
    cosmologies         = pd.read_csv(cosmologies_file)

    ## Get the index of the version
    ## The csv contains much whitespace, so we extract the version number from the column names
    colnames            = cosmologies.columns # Column names
    sim_names_colname   = colnames[0]   # First column is the simulation names
    sim_names           = cosmologies[sim_names_colname].values # List of simulation names: abacus_cosm_vvv
    sim_names_versions  = [int(sim_name[11:]) for sim_name in sim_names] # Version number of each simulation
    idx                 = np.where(np.array(sim_names_versions) == version)[0][0] # Index of the version we want
    
    sim_params          = cosmologies.iloc[idx]#[2:] # LCDM parameters

    # Check that the version number in the csv file matches the version argument 
    version_number_csv = int(sim_params.iloc[0].strip()[-3:])
    if version_number_csv != version:
        print(f"Error: version {version} does not match version in csv file {version_number_csv}")
        exit()
    
    if param_name is None:
        print("param_name is None")
        # Return all parameters
        return sim_params
    
    else:
        if param_name not in csv_column_names:
            print(f"Error: parameter {param_name} not found in csv_column_names")
            exit()
        for i in range(2, len(sim_params)):
            key = sim_params.index[i].strip()
            if key == param_name:
                return sim_params.iloc[i]

    print(f"Error: Something went wrong.")
    exit()


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
    As          = get_sim_params_from_csv_table(version=version, param_name="A_s")
    ln1e10As    = np.log(As * 1.0e10)
    ns          = header['n_s']
    w           = header['w']
    w0          = header["w0"]
    wa          = header["wa"]
    sigma8      = get_sim_params_from_csv_table(version=version, param_name="sigma8_m") 
    Om          = header['Omega_M']
    h           = header['H0'] / 100.0     
    N_eff       = header['N_ncdm'] + header['N_ur']

    df = pd.DataFrame({
        'version'   : version,
        'redshift'  : redshift,
        'wb'        : wb,
        'wc'        : wc,
        'Ol'        : Ol,
        'ln1e10As'  : ln1e10As,
        'ns'        : ns,
        'w'         : w,
        'w0'        : w0,
        'wa'        : wa,
        'sigma8'    : sigma8,
        'Om'        : Om,
        'h'         : h,
        'N_eff'     : N_eff
    }, index=[0])

    # Store df in each phase directory
    print(f"Saving {OUTFILENAME} for all phases of c000.")
    for ph in range(0, 25):
        VERSIONPATH = Path(D13_EMULATION_PATH / f"AbacusSummit_base_c000_ph{str(ph).zfill(3)}")
        VERSIONPATH.mkdir(parents=True, exist_ok=True)

        paramfile = Path(VERSIONPATH / OUTFILENAME)
        if paramfile.exists():
            print(f"Error: file {OUTFILENAME} already exists. Skipping.")
            continue

        df.to_csv(paramfile, index=False, sep=" ")

    print("Finished")


def save_cosmo_parameters_all_versions():
    versions = np.arange(0,5)
    versions = np.concatenate((versions, np.arange(100, 127)))
    versions = np.concatenate((versions, np.arange(130,182)))
    print("Saving cosmological parameters for versions: ")
    for ver in versions:
        print(f"{ver}", end=", ")
        VERSIONPATH = Path(D13_EMULATION_PATH / f"AbacusSummit_base_c{str(ver).zfill(3)}_ph000")
        VERSIONPATH.mkdir(parents=True, exist_ok=True)
        paramfile = Path(VERSIONPATH / OUTFILENAME)
        if paramfile.exists():
            print(f"Error: file {OUTFILENAME} already exists. Skipping.")
            continue
        
        header      = get_asdf_version_header(version=ver)
        redshift    = header['Redshift']
        wb          = header['omega_b']
        wc          = header['omega_cdm']
        Ol          = header['Omega_DE']
        As          = get_sim_params_from_csv_table(version=ver, param_name="A_s")
        ln1e10As    = np.log(As * 1.0e10)
        ns          = header['n_s']
        w           = header['w']
        w0          = header["w0"]
        wa          = header["wa"]
        sigma8      = get_sim_params_from_csv_table(version=ver, param_name="sigma8_m") 
        Om          = header['Omega_M']
        h           = header['H0'] / 100.0     
        N_eff       = header['N_ncdm'] + header['N_ur']

        df = pd.DataFrame({
            'version'   : ver,
            'redshift'  : redshift,
            'wb'        : wb,
            'wc'        : wc,
            'Ol'        : Ol,
            'ln1e10As'  : ln1e10As,
            'ns'        : ns,
            'w'         : w,
            'w0'        : w0,
            'wa'        : wa,
            'sigma8'    : sigma8,
            'Om'        : Om,
            'h'         : h,
            'N_eff'     : N_eff
        }, index=[0])
        df.to_csv(paramfile, index=False, sep=" ")

    print("Finished")

save_cosmo_parameters_c000_all_phases()
save_cosmo_parameters_all_versions()



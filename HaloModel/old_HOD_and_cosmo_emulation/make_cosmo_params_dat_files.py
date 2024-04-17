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
OUTFILENAME         = Path(f"cosmological_parameters.dat") # Same filename for all simulations

CSV_COLUMN_NAMES = ['omega_b', 'omega_cdm', 'h', 
                    'A_s', 'n_s', 'alpha_s', 
                    'N_ur', 'N_ncdm', 'omega_ncdm', 
                    'w0_fld', 'wa_fld', 'sigma8_m', 
                    'sigma8_cb']

def get_asdf_version_header(
        version: int = 130,
        phase:   int = 0,
        ):
    """
    Return the header of the asdf file for a given version and file number.
    Contains some of the cosmological parameters we want to save 
    """

    # Same header for all files of a simulation, load the first one 
    simname     = Path(f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}")
    asdf_fname  = Path(D13_BASE_PATH / simname / "halos/z0.250/halo_info/halo_info_000.asdf")


    if not asdf_fname.exists():        
        raise FileNotFoundError(f"Error: file {asdf_fname} does not exist")
    
    # Load the header, and return it 
    af = asdf.open(asdf_fname)
    header = af['header']
    af.close()
    return header 


def get_sim_params_from_csv_table(
        version:    int = 130,
        param_name: Optional[str] = None
        ):
    
    """
    Some simulation/cosmological parameters are not stored in the header files.
    These are given here: https://abacussummit.readthedocs.io/en/latest/cosmologies.html
    The .csv file downloaded contains these parameters for all simulation versions.

    param_name: Optional[str].
    If None, return all parameters as a pandas Series.
    Otherwise, param_name has to be one of the elements in CSV_COLUMN_NAMES.
    Then, the parameter value is returned as a float.
    """

    cosmologies_file    = Path(D13_BASE_PATH / "cosmologies.csv")
    cosmologies         = pd.read_csv(cosmologies_file, index_col=0)

    # Format the version number to match the csv file     
    sim_name            = f"abacus_cosm{str(version).zfill(3)}"
    csv_sim_names       = cosmologies.index.str.strip().values # Strip whitespace from sim-version column names
    idx                 = np.where(csv_sim_names == sim_name)[0][0] # Row index of the version we want
    sim_params          = cosmologies.iloc[idx] # Pandas Series with all parameters for this version


    if param_name is None:
        print("param_name is None")
        # Return all parameters
        return sim_params
    
    else:
        # Return the value of param_name  
        if param_name not in CSV_COLUMN_NAMES:
            print(f"Error: parameter {param_name} not found in csv_column_names")
            exit()
        for i in range(2, len(sim_params)):
            key = sim_params.index[i].strip()
            if key == param_name:
                return sim_params.iloc[i]


def get_cosmo_parameters_df(
        version:   int = 130,
        ):
    
    header      = get_asdf_version_header(version=version)
    redshift    = header['Redshift']
    wb          = header['omega_b']
    wc          = header['omega_cdm']
    Ol          = header['Omega_DE']
    As          = get_sim_params_from_csv_table(version=version, param_name="A_s")
    ln1e10As    = np.log(As * 1.0e10)
    ns          = header['n_s']
    alpha_s     = get_sim_params_from_csv_table(version=version, param_name="alpha_s")
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
        'alpha_s'   : alpha_s,
        'w'         : w,
        'w0'        : w0,
        'wa'        : wa,
        'sigma8'    : sigma8,
        'Om'        : Om,
        'h'         : h,
        'N_eff'     : N_eff
    }, index=[0])

    return df 
    


def save_cosmo_parameters_c000_all_phases():
    """
    All phases have the same parameters. 
    Get the parameters from the first phase.
    Store copies of the file in the directory of each phase.
    """
    df = get_cosmo_parameters_df(version=0)

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
        
        df = get_cosmo_parameters_df(version=ver)
        df.to_csv(paramfile, index=False, sep=" ")

    print("Finished")

# save_cosmo_parameters_c000_all_phases()
# save_cosmo_parameters_all_versions()




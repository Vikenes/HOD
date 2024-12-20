from pathlib import Path
import yaml
from HOD_and_cosmo_prior_ranges import get_HOD_params_prior_range, get_cosmo_params_prior_range
"""
Creates yaml file of prior ranges of HOD and cosmological parameters.
"""

OUTPATH             = Path("/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data")
HOD_PARAM_NAMES     = ["log10M1", "sigma_logM", "kappa", "alpha", "log10_ng"]
COSMO_PARAM_NAMES   = ['N_eff', 'alpha_s', 'ns', 'sigma8', 'w0', 'wa', 'wb', 'wc']

"""
Generate a yaml file with the prior ranges of the HOD and cosmological parameters.
This file is used in the inference code to set the prior ranges of the parameters.
"""

def make_priors_config_file():

    hod   = get_HOD_params_prior_range()
    cosmo = get_cosmo_params_prior_range()

    hod     = {key: [float(hod[key][0]), float(hod[key][1])] for key in HOD_PARAM_NAMES}
    cosmo   = {key: [float(cosmo[key][0]), float(cosmo[key][1])] for key in COSMO_PARAM_NAMES}
    priors  = hod | cosmo
    outfile = Path(OUTPATH / "priors_config.yaml")
    if outfile.exists():
        print(f"File {outfile} already exists.")
        return
    else:
        print(f"Storing priors in {outfile}.")
        with open(outfile, "w") as f:
            yaml.dump(priors, f, default_flow_style=False)
    
make_priors_config_file()
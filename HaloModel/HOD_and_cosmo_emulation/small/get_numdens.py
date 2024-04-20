import h5py 
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd 
import time 
import concurrent.futures
import sys

sys.path.append("../")
from numdens_HOD import get_log10Mmin_from_fixed_log_ng_small

BASE_PATH = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/small")

    
def print_logMmin():
    log10Mmin_array = np.load("log10Mmin_small.npy")
    mean = np.mean(log10Mmin_array)
    std  = np.std(log10Mmin_array)
    print(f"Log10Mmin for {len(log10Mmin_array)} AbacusSummit_small_c000_ph*:")
    print(f"    Mean:    {mean:.4f}")
    print(f"    Stddev:  {std:.4e}")
    print(f"    Min:     {np.min(log10Mmin_array):.4f}")   
    print(f"    Max:     {np.max(log10Mmin_array):.4f}")   
    print(f"    Max-Min: {np.max(log10Mmin_array)-np.min(log10Mmin_array):.5f}")   





def store_log10Mmin_small():
    HOD_PARAMS_PATH  = f"/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files"
    HOD_PARAMS       = pd.read_csv(
        f"{HOD_PARAMS_PATH}/AbacusSummit_base_c000_ph000/HOD_parameters/HOD_parameters_fiducial.csv"
        )
    
    outfile = Path("./log10Mmin_small.npy")
    if outfile.exists():
        raise FileExistsError(f"File {outfile.name} already exists.")
    
    all_phases = np.arange(3000, 5000)
    mass_array_paths = np.array([Path(BASE_PATH / f"AbacusSummit_small_c000_ph{phase}/pos_vel_mass_arrays") for phase in all_phases if Path(BASE_PATH / f"AbacusSummit_small_c000_ph{phase}").is_dir()])
    mass_array_paths = mass_array_paths

    sigma_logM  = HOD_PARAMS["sigma_logM"][0]#0.6915 
    log10M1     = HOD_PARAMS["log10M1"][0]#14.42 # h^-1 Msun
    kappa       = HOD_PARAMS["kappa"][0]#0.51 
    alpha       = HOD_PARAMS["alpha"][0]#0.9168
    log_ng      = HOD_PARAMS["log10_ng"][0]#-3.45  

    params = (sigma_logM, log10M1, kappa, alpha, log_ng)

    log10Mmin_array = np.zeros_like(mass_array_paths, dtype=np.float64)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(get_log10Mmin_from_fixed_log_ng_small, path, params) for path in mass_array_paths]
        for i, f in enumerate(concurrent.futures.as_completed(results)):
            # Add float value of f to list 
            log10Mmin_array[i] = f.result().item()    

    np.save(outfile, log10Mmin_array)

# store_log10Mmin_small()
print_logMmin()

import numpy as np 
import h5py 
import time 
import matplotlib.pyplot as plt
import pandas as pd 
from pathlib import Path


D13_BASE_PATH = "/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files"
D13_EMULATION_PATH  = Path(D13_BASE_PATH)
def get_sim_params_from_csv_table(
        version:    int = 130,
        param_name: str = None
        ):
    dat_path = Path(D13_EMULATION_PATH / f"AbacusSummit_base_c{str(version).zfill(3)}_ph000/cosmological_parameters.dat")
    csv_path = Path(f"/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/cosmologies.csv")

    """
    Some simulation/cosmological parameters are not stored in the header files.
    These are given here: https://abacussummit.readthedocs.io/en/latest/cosmologies.html
    The .csv file downloaded contains these parameters for all simulation versions.

    param_name: Optional[str].
    If None, return all parameters as a pandas Series.
    Otherwise, param_name has to be one of the elements in CSV_COLUMN_NAMES.
    Then, the parameter value is returned as a float.
    """

    cosmo_dict = pd.read_csv(
        dat_path,
        sep=" "
        ).iloc[0].to_dict()

    cosmologies         = pd.read_csv(csv_path, index_col=0)
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
        for i in range(2, len(sim_params)):
            key = sim_params.index[i].strip()
            if key == param_name:
                return sim_params.iloc[i]


def get_version_path_list(
    version_low:  int  = 0, 
    version_high: int  = 181,
    phase:        bool = False,
    ):
    get_sim_path = lambda version, phase: Path(D13_EMULATION_PATH / f"AbacusSummit_base_c{str(version).zfill(3)}_ph{str(phase).zfill(3)}")
    if phase:
        version_list = [get_sim_path(0, ph) for ph in range(25) if get_sim_path(0, ph).is_dir()]
    else:
        version_list = [get_sim_path(v, 0) for v in range(version_low, version_high) if get_sim_path(v, 0).is_dir()]
    return version_list

C000_PATHS              = get_version_path_list(phase=True)
C001_C004_PATHS         = get_version_path_list(version_low=1, version_high=5, phase=False)
LIN_DER_GRID_PATHS      = get_version_path_list(version_low=100, version_high=127, phase=False)
BROAD_EMUL_GRID_PATHS   = get_version_path_list(version_low=130, version_high=182, phase=False)
# All simulations
HOD_PATHS        = C000_PATHS + C001_C004_PATHS + LIN_DER_GRID_PATHS + BROAD_EMUL_GRID_PATHS
N_PATHS          = len(HOD_PATHS)
BOX_SIZE         = 2000.0**3
dset_sizes = {"train": 500, "test": 100, "val": 100}



def store_ng_array(flag: str = "test"):

    ng_array         = np.zeros(N_PATHS * dset_sizes[flag])
    for i, path in enumerate(HOD_PATHS):
        HOD_cat = h5py.File(path / f"HOD_catalogues/halocat_{flag}_ng_fixed.hdf5", 'r')
        for j, node in enumerate(HOD_cat.keys()):
            ng_array[i*dset_sizes[flag] + j] = HOD_cat[node]['x'].shape[0]
        print(f"completed {i+1}/{N_PATHS} simulations.")
        HOD_cat.close()
    
    ng_array /= BOX_SIZE
    np.save(f"ng_array_{flag}.npy", ng_array)

# store_ng_array("test")
# store_ng_array("train")
# store_ng_array("val")
# exit()

def load_ng_array_c000_c004(flag: str = "test"):
    start = 0
    stop = (len(C000_PATHS) + len(C001_C004_PATHS)) * dset_sizes[flag]
    # print(stop)
    # print(f"{flag=}")
    ng_array = np.load(f"ng_array_{flag}.npy")
    # print(ng_array[-10:])
    # print(ng_array.shape)
    # exit()
    ng_array = ng_array[start : start + stop]
    x = np.linspace(0, len(C000_PATHS) + len(C001_C004_PATHS), len(ng_array))

    plt.plot(x, ng_array, "o", ms=2)
    plt.hlines(2.174e-4, x[0], x[-1], linestyle="dashed", colors='k', label=r"$n_g$ desired")
    plt.legend()
    plt.yscale("log")
    plt.show()

def load_ng_array_c000(flag: str = "test"):
    start = 0
    stop = len(C000_PATHS) * dset_sizes[flag]
    # print(stop)
    # print(f"{flag=}")
    ng_array = np.load(f"ng_array_{flag}.npy")
    # print(ng_array[-10:])
    # print(ng_array.shape)
    # exit()
    ng_array = ng_array[start : start + stop]
    x = np.linspace(0, len(C000_PATHS) + len(C001_C004_PATHS), len(ng_array))

    plt.plot(x, ng_array, "o", ms=2)
    plt.hlines(2.174e-4, x[0], x[-1], linestyle="dashed", colors='k', label=r"$n_g$ desired")
    plt.legend()
    plt.yscale("log")
    plt.show()


def load_ng_array_c004(flag: str = "test"):
    start = len(C000_PATHS) * dset_sizes[flag]
    stop = len(C001_C004_PATHS) * dset_sizes[flag]
    # print(stop)
    # print(f"{flag=}")
    ng_array = np.load(f"ng_array_{flag}.npy")
    # print(ng_array[-10:])
    # print(ng_array.shape)
    # exit()
    ng_array = ng_array[start : start + stop]
    # Compute mean of ng_array over intervals of 100
    ng_array_mean = np.mean(ng_array.reshape(-1, 100), axis=1)
    for i in range(4):
        s8 = get_sim_params_from_csv_table(version=i+1, param_name="sigma8_m")
        print(f"{ng_array_mean[i] / s8=:.5e}")
    exit()
    x = np.linspace(1, 5, len(ng_array))

    plt.plot(x, ng_array, "o", ms=2)
    plt.hlines(2.174e-4, x[0], x[-1], linestyle="dashed", colors='k', label=r"$n_g$ desired")
    plt.legend()
    plt.yscale("log")
    plt.show()

def load_ng_array_c100_c126(flag: str = "test"):
    start = (len(C000_PATHS) + len(C001_C004_PATHS)) * dset_sizes[flag]
    stop = len(LIN_DER_GRID_PATHS) * dset_sizes[flag]

    ng_array = np.load(f"ng_array_{flag}.npy") 
    ng_array = ng_array[start : start + stop]
    # print(np.argmax(ng_array)/dset_sizes[flag])
    x = np.linspace(100, 127, len(ng_array)) 
    # print(x[np.argmax(ng_array)])
    # exit()

    plt.plot(x, ng_array, "o", ms=2)
    plt.hlines(2.174e-4, x[0], x[-1], linestyle="dashed", colors='k', label=r"$n_g$ desired")
    plt.legend()
    plt.yscale("log")
    plt.show()
 

 
def load_ng_array_c130_c181(flag: str = "test"):
    start = (len(C000_PATHS) + len(C001_C004_PATHS) + len(LIN_DER_GRID_PATHS)) * dset_sizes[flag]

    ng_array = np.load(f"ng_array_{flag}.npy") # [dset_sizes[flag] * start:]
    ng_array = ng_array[start : ]
    x = np.linspace(130, 182, len(ng_array)) # / dset_sizes[flag]

    plt.plot(x, ng_array, "o", ms=2)
    plt.hlines(2.174e-4, x[0], x[-1], linestyle="dashed", colors='k', label=r"$n_g$ desired")
    plt.legend()
    plt.yscale("log")
    plt.show()



def plot(flag: str = "test"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    start0 = 0
    start1 = (len(C000_PATHS) + len(C001_C004_PATHS)) * dset_sizes[flag]
    start2 = (len(C000_PATHS) + len(C001_C004_PATHS) + len(LIN_DER_GRID_PATHS)) * dset_sizes[flag]

    stop0 = start0 + (len(C000_PATHS) + len(C001_C004_PATHS)) * dset_sizes[flag]
    stop1 = start1 + len(LIN_DER_GRID_PATHS) * dset_sizes[flag]
    stop2 = start2 + len(BROAD_EMUL_GRID_PATHS) * dset_sizes[flag]

    ng_array = np.load(f"ng_array_{flag}.npy") # [dset_sizes[flag] * start:]
    ng_array0 = ng_array[start0 : stop0]
    ng_array1 = ng_array[start1 : stop1]
    ng_array2 = ng_array[start2 : stop2]
    x0 = np.linspace(0, len(C000_PATHS) + len(C001_C004_PATHS), len(ng_array0))
    x1 = np.linspace(100, 127, len(ng_array1))
    x2 = np.linspace(130, 182, len(ng_array2))
    fig.suptitle(rf"$n_g$ from catalogues for dataset {flag}")
    ax[0].set_title("c000 - c004")
    ax[1].set_title("c100 - c126")
    ax[2].set_title("c130 - c181")
    ax[0].plot(x0, ng_array0, "o", ms=2)
    ax[1].plot(x1, ng_array1, "o", ms=2)
    ax[2].plot(x2, ng_array2, "o", ms=2)
    ax[0].hlines(2.174e-4, x0[0], x0[-1], linestyle="dashed", colors='k', label=r"$n_g$ desired")
    ax[1].hlines(2.174e-4, x1[0], x1[-1], linestyle="dashed", colors='k', label=r"$n_g$ desired")
    ax[2].hlines(2.174e-4, x2[0], x2[-1], linestyle="dashed", colors='k', label=r"$n_g$ desired")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    
    # Increase horizontal spacing between  figure that's being saved
    plt.subplots_adjust(wspace=0.3)
    # fig.savefig(f"ng_array_{flag}.png", dpi=300, bbox_inches="tight")
    plt.show()

# print(HOD_PATHS)
# exit()
# load_ng_array_c000_c004("val")
# load_ng_array_c100_c126("val")
load_ng_array_c004("val")
# load_ng_array_c000("val")

# load_ng_array_c130_c181("val")
# plot("val")